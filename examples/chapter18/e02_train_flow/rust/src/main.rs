use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, DatasetError};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d,
};
use burn::optim::AdamWConfig;
use burn::tensor::activation::silu;
use burn::tensor::{Device, Int, Tensor, TensorData};
use burn::train::logger::{EvaluationProgressLogger, TrainingProgressLogger};
use burn::train::metric::{LossMetric, MetricDefinition};
use burn::train::renderer::{
    EvaluationName, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining,
};
use burn::train::{
    InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};
use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot, SafetensorsStore};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SIZE: usize = 12;
const PIXELS: usize = SIZE * SIZE;
const NULL_CLASS: usize = 4; // the "no label" class used for classifier-free guidance

// --- the data: Chapter 13's 12x12 shapes, generated in code (identical in python/main.py) ---
// A tiny LCG so Rust and Python draw exactly the same "random" numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    fn below(&mut self, n: u32) -> usize {
        (self.next() % n) as usize
    }
    fn uniform(&mut self) -> f32 {
        (self.next() as f32 + 0.5) / 32768.0 // in (0, 1), exact in f32
    }
    // Box-Muller: two uniforms -> two standard normals (computed in f64, stored as f32)
    fn normals(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let u1 = (self.next() as f64 + 1.0) / 32768.0;
            let u2 = self.next() as f64 / 32768.0;
            let r = (-2.0 * u1.ln()).sqrt();
            let a = 2.0 * std::f64::consts::PI * u2;
            out.push((r * a.cos()) as f32);
            out.push((r * a.sin()) as f32);
        }
        out.truncate(n);
        out
    }
}

// class 0 = horizontal bar, 1 = vertical bar, 2 = diagonal, 3 = plus sign
fn make_image(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut img = vec![0.0f32; PIXELS];
    match class {
        0 => {
            let (r, c) = (rng.below(12), rng.below(7));
            (0..6).for_each(|j| img[r * SIZE + c + j] = 1.0);
        }
        1 => {
            let (c, r) = (rng.below(12), rng.below(7));
            (0..6).for_each(|i| img[(r + i) * SIZE + c] = 1.0);
        }
        2 => {
            let (r, c) = (rng.below(7), rng.below(7));
            (0..6).for_each(|i| img[(r + i) * SIZE + c + i] = 1.0);
        }
        _ => {
            let (r, c) = (2 + rng.below(8), 2 + rng.below(8));
            for d in 0..5 {
                img[r * SIZE + c + d - 2] = 1.0;
                img[(r + d - 2) * SIZE + c] = 1.0;
            }
        }
    }
    // background noise in [0, 0.25): exact in f32, so both languages agree bit for bit
    img.iter_mut()
        .for_each(|p| *p += rng.next() as f32 / 32768.0 * 0.25);
    img
}

// Pixels in [0, 1.25) -> [-1, 1): the range the noise lives in.
fn to_signed(img: Vec<f32>) -> Vec<f32> {
    img.into_iter().map(|p| 1.6 * p - 1.0).collect()
}

// One training example of flow matching: data x1, noise x0, a time t, a label.
#[derive(Clone, Debug)]
struct FlowItem {
    x1: Vec<f32>,
    x0: Vec<f32>,
    t: f32,
    label: usize,
}

// Item `index` is a pure function of (seed, index): a fresh image, fresh noise and a fresh t.
fn flow_item(seed: u32, index: usize, p_drop: u32) -> FlowItem {
    let mut rng = Lcg(seed ^ (index as u32).wrapping_mul(2654435761));
    let class = index % 4;
    let x1 = to_signed(make_image(class, &mut rng));
    let x0 = rng.normals(PIXELS);
    let t = rng.uniform();
    let drop = rng.below(100) < p_drop as usize; // p_drop percent of labels -> "no label"
    FlowItem {
        x1,
        x0,
        t,
        label: if drop { NULL_CLASS } else { class },
    }
}

// --- the time embedding: sin/cos of t at 16 frequencies, as in a transformer ---
fn time_embedding(t: Tensor<1>, dim: usize) -> Tensor<2> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::<1>::from_floats(freqs.as_slice(), &t.device()).unsqueeze_dim(0); // [1, half]
    let args = t.unsqueeze_dim(1).mul_scalar(1000.0) * freqs; // [N, half]
    Tensor::cat(vec![args.clone().sin(), args.cos()], 1) // [N, dim]
}

// --- the U-Net ---
// A residual block that also reads the conditioning vector c (time + class).
#[derive(Module, Debug)]
struct ResBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    cond: Linear, // c -> one shift per output channel
    norm2: GroupNorm,
    conv2: Conv2d,
    skip: Option<Conv2d>, // 1x1 conv when the channel count changes
}

impl ResBlock {
    fn new(c_in: usize, c_out: usize, d_cond: usize, groups: usize, device: &Device) -> Self {
        let conv = |a, b| {
            Conv2dConfig::new([a, b], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        Self {
            norm1: GroupNormConfig::new(groups, c_in).init(device),
            conv1: conv(c_in, c_out),
            cond: LinearConfig::new(d_cond, c_out).init(device),
            norm2: GroupNormConfig::new(groups, c_out).init(device),
            conv2: conv(c_out, c_out),
            skip: (c_in != c_out).then(|| Conv2dConfig::new([c_in, c_out], [1, 1]).init(device)),
        }
    }

    fn forward(&self, x: Tensor<4>, c: Tensor<2>) -> Tensor<4> {
        let h = self.conv1.forward(silu(self.norm1.forward(x.clone())));
        let shift = self.cond.forward(silu(c)).unsqueeze_dims::<4>(&[2, 3]); // [N, C, 1, 1]
        let h = self.conv2.forward(silu(self.norm2.forward(h + shift)));
        let skip = match &self.skip {
            Some(conv) => conv.forward(x),
            None => x,
        };
        h + skip
    }
}

#[derive(Module, Debug)]
struct UNet {
    time1: Linear,
    time2: Linear,
    class_emb: Embedding, // 4 classes + the null class
    conv_in: Conv2d,
    enc: ResBlock,       //  C channels at 12x12 -> the skip connection
    down: Conv2d,        //  C -> 2C, stride 2: 12x12 -> 6x6
    mid: ResBlock,       // 2C channels at 6x6
    up: ConvTranspose2d, // 2C -> C, stride 2: 6x6 -> 12x12
    dec: ResBlock,       // [up, skip] = 2C -> C
    norm_out: GroupNorm,
    conv_out: Conv2d, // C -> 1: the velocity
    t_dim: usize,
}

impl UNet {
    fn new(c: usize, t_dim: usize, d: usize, device: &Device) -> Self {
        let g = 4; // GroupNorm groups
        let same = || PaddingConfig2d::Explicit(1, 1, 1, 1);
        Self {
            time1: LinearConfig::new(t_dim, d).init(device),
            time2: LinearConfig::new(d, d).init(device),
            class_emb: EmbeddingConfig::new(5, d).init(device),
            conv_in: Conv2dConfig::new([1, c], [3, 3])
                .with_padding(same())
                .init(device),
            enc: ResBlock::new(c, c, d, g, device),
            down: Conv2dConfig::new([c, 2 * c], [3, 3])
                .with_stride([2, 2])
                .with_padding(same())
                .init(device),
            mid: ResBlock::new(2 * c, 2 * c, d, g, device),
            up: ConvTranspose2dConfig::new([2 * c, c], [2, 2])
                .with_stride([2, 2])
                .init(device),
            dec: ResBlock::new(2 * c, c, d, g, device),
            norm_out: GroupNormConfig::new(g, c).init(device),
            conv_out: Conv2dConfig::new([c, 1], [3, 3])
                .with_padding(same())
                .init(device),
            t_dim,
        }
    }

    /// The velocity field v(x, t, y). x: [N, 1, 12, 12], t: [N], labels: [N] (4 = no label).
    fn forward(&self, x: Tensor<4>, t: Tensor<1>, labels: Tensor<1, Int>) -> Tensor<4> {
        let [n] = labels.dims();
        let temb = self
            .time2
            .forward(silu(self.time1.forward(time_embedding(t, self.t_dim))));
        let cemb = self
            .class_emb
            .forward(labels.reshape([n, 1]))
            .reshape([n, temb.dims()[1]]);
        let c = temb + cemb; // one conditioning vector per image

        let h = self.conv_in.forward(x);
        let skip = self.enc.forward(h, c.clone()); // [N, C, 12, 12]
        let h = self.mid.forward(self.down.forward(skip.clone()), c.clone()); // [N, 2C, 6, 6]
        let h = Tensor::cat(vec![self.up.forward(h), skip], 1); // [N, 2C, 12, 12]
        let h = self.dec.forward(h, c);
        self.conv_out.forward(silu(self.norm_out.forward(h)))
    }
}

// --- fixed starting weights, so every number can be checked against PyTorch ---
// Chapter 13's formula, w_i = sin(2.3 i + phase) / sqrt(fan_in), with a new phase per tensor.
// Biases start at zero; GroupNorm keeps its default gamma = 1, beta = 0.
fn det(n: usize, fan_in: usize, phase: f64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f64 * 2.3 + phase).sin() / (fan_in as f64).sqrt()) as f32)
        .collect()
}

fn param<const D: usize>(v: Vec<f32>, shape: [usize; D], device: &Device) -> Param<Tensor<D>> {
    Param::from_tensor(Tensor::from_data(TensorData::new(v, shape), device))
}

struct Fixed<'a> {
    k: usize, // tensors filled so far: the phase is 0.1 * k
    device: &'a Device,
}

impl Fixed<'_> {
    fn phase(&mut self) -> f64 {
        self.k += 1;
        0.1 * self.k as f64
    }
    fn conv(&mut self, conv: &mut Conv2d) {
        let [o, i, kh, kw] = conv.weight.val().dims();
        let w = det(o * i * kh * kw, i * kh * kw, self.phase());
        conv.weight = param(w, [o, i, kh, kw], self.device);
        conv.bias = Some(param(vec![0.0; o], [o], self.device));
    }
    fn conv_t(&mut self, conv: &mut ConvTranspose2d) {
        let [i, o, kh, kw] = conv.weight.val().dims(); // [in, out, k, k], as in PyTorch
        conv.weight = param(
            det(i * o * kh * kw, i, self.phase()),
            [i, o, kh, kw],
            self.device,
        );
        conv.bias = Some(param(vec![0.0; o], [o], self.device));
    }
    fn linear(&mut self, lin: &mut Linear) {
        let [i, o] = lin.weight.val().dims(); // Burn stores [in, out]; PyTorch [out, in]
        let w = det(o * i, i, self.phase()); // generated in PyTorch's order...
        let w_t: Vec<f32> = (0..i * o).map(|k| w[(k % o) * i + k / o]).collect(); // ...transposed
        lin.weight = param(w_t, [i, o], self.device);
        lin.bias = Some(param(vec![0.0; o], [o], self.device));
    }
    fn block(&mut self, b: &mut ResBlock) {
        self.conv(&mut b.conv1);
        self.linear(&mut b.cond);
        self.conv(&mut b.conv2);
        if let Some(s) = b.skip.as_mut() {
            self.conv(s);
        }
    }
}

impl UNet {
    fn with_fixed_weights(mut self, device: &Device) -> Self {
        let mut f = Fixed { k: 0, device };
        f.linear(&mut self.time1);
        f.linear(&mut self.time2);
        let [n, d] = self.class_emb.weight.val().dims();
        self.class_emb.weight = param(det(n * d, 1, f.phase()), [n, d], device);
        f.conv(&mut self.conv_in);
        f.block(&mut self.enc);
        f.conv(&mut self.down);
        f.block(&mut self.mid);
        f.conv_t(&mut self.up);
        f.block(&mut self.dec);
        f.conv(&mut self.conv_out);
        self
    }
}

// --- the data pipeline: a Dataset whose items are computed, not stored (Chapter 12) ---
// Item i is a fresh image, fresh noise, a fresh t and a label that is dropped p_drop% of
// the time -- the same numbers in Rust and Python, and never the same pair twice.
struct FlowDataset {
    seed: u32,
    len: usize,
    p_drop: u32,
}

impl Dataset<FlowItem> for FlowDataset {
    fn get(&self, index: usize) -> Result<FlowItem, DatasetError> {
        assert!(index < self.len, "index {index} out of range");
        Ok(flow_item(self.seed, index, self.p_drop))
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone, Debug)]
struct FlowBatch {
    x1: Tensor<4>,          // data      [N, 1, 12, 12]
    x0: Tensor<4>,          // noise     [N, 1, 12, 12]
    t: Tensor<1>,           // times     [N]
    labels: Tensor<1, Int>, // classes   [N], 4 = no label
}

#[derive(Clone, Default)]
struct FlowBatcher;

impl Batcher<FlowItem, FlowBatch> for FlowBatcher {
    fn batch(&self, items: Vec<FlowItem>, device: &Device) -> FlowBatch {
        let n = items.len();
        let x1: Vec<f32> = items.iter().flat_map(|it| it.x1.clone()).collect();
        let x0: Vec<f32> = items.iter().flat_map(|it| it.x0.clone()).collect();
        let t: Vec<f32> = items.iter().map(|it| it.t).collect();
        let labels: Vec<i64> = items.iter().map(|it| it.label as i64).collect();
        FlowBatch {
            x1: Tensor::from_data(TensorData::new(x1, [n, 1, SIZE, SIZE]), device),
            x0: Tensor::from_data(TensorData::new(x0, [n, 1, SIZE, SIZE]), device),
            t: Tensor::from_data(TensorData::new(t, [n]), device),
            labels: Tensor::from_data(TensorData::new(labels, [n]), device),
        }
    }
}

// --- the model's config, and the run's ---
#[derive(Config, Debug)]
struct UNetConfig {
    #[config(default = 16)]
    channels: usize,
    #[config(default = 32)]
    t_dim: usize,
    #[config(default = 64)]
    d_cond: usize,
}

impl UNetConfig {
    fn init(&self, device: &Device) -> UNet {
        UNet::new(self.channels, self.t_dim, self.d_cond, device)
    }
}

#[derive(Config, Debug)]
struct TrainingConfig {
    model: UNetConfig,
    optimizer: AdamWConfig,
    #[config(default = 1200)]
    steps: usize,
    #[config(default = 64)]
    batch_size: usize,
    #[config(default = 3e-3)]
    learning_rate: f64,
    #[config(default = 10)]
    p_drop: u32, // percent of labels replaced by the null class
    #[config(default = 18)]
    seed: u32,
}

// --- the flow-matching objective, and the two traits the Learner needs ---
impl UNet {
    fn forward_flow(&self, batch: FlowBatch) -> RegressionOutput {
        let n = batch.t.dims()[0];
        let t = batch.t.clone().unsqueeze_dims::<4>(&[1, 2, 3]); // [N, 1, 1, 1]
        let xt = batch.x0.clone() * (t.clone().neg() + 1.0) + batch.x1.clone() * t; // straight path
        let target = batch.x1 - batch.x0; // v = x1 - x0, the same for every t
        let v = self.forward(xt, batch.t, batch.labels);
        let loss = (v.clone() - target.clone()).powi_scalar(2).mean(); // mean over N * 144
        RegressionOutput::new(loss, v.reshape([n, PIXELS]), target.reshape([n, PIXELS]))
    }
}

impl TrainStep for UNet {
    type Input = FlowBatch;
    type Output = RegressionOutput;

    fn step(&self, batch: FlowBatch) -> TrainOutput<RegressionOutput> {
        let item = self.forward_flow(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for UNet {
    type Input = FlowBatch;
    type Output = RegressionOutput;

    fn step(&self, batch: FlowBatch) -> RegressionOutput {
        self.forward_flow(batch)
    }
}

// A renderer that draws nothing (Chapter 14): we read the Learner's log files instead.
struct Quiet;
impl MetricsRendererTraining for Quiet {
    fn update_train(&mut self, _: MetricState) {}
    fn update_valid(&mut self, _: MetricState) {}
}
impl TrainingProgressLogger for Quiet {
    fn start(&mut self, _: usize, _: usize, _: Option<usize>, _: Option<&str>) {}
    fn update_epoch(&mut self, _: usize) {}
    fn start_split(&mut self, _: &str, _: usize) {}
    fn update_split(&mut self, _: usize) {}
    fn end_split(&mut self) {}
    fn end(&mut self) {}
    fn log_event_training(&mut self, _: String) {}
}
impl MetricsRendererEvaluation for Quiet {
    fn update_test(&mut self, _: EvaluationName, _: MetricState) {}
}
impl EvaluationProgressLogger for Quiet {
    fn start_global_progress(&mut self, _: usize) {}
    fn start_test(&mut self, _: &str, _: usize) {}
    fn update_test_progress(&mut self, _: usize) {}
    fn end_test(&mut self) {}
    fn end_global_progress(&mut self) {}
    fn log_event_evaluation(&mut self, _: String) {}
}
impl MetricsRenderer for Quiet {
    fn manual_close(&mut self) {}
    fn register_metric(&mut self, _: MetricDefinition) {}
}

// Every batch's loss, in order, from <dir>/<split>/epoch-1/Loss.log ("value,count" lines).
fn batch_losses(dir: &Path, split: &str) -> Vec<f64> {
    let log = std::fs::read_to_string(dir.join(format!("{split}/epoch-1/Loss.log"))).unwrap();
    log.lines()
        .filter_map(|line| line.split_once(','))
        .filter(|(_, c)| *c != "final")
        .map(|(v, _)| v.parse().unwrap())
        .collect()
}

fn weights_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights")
}

/// Trains for `steps` batches with the Learner, one epoch of fresh data.
/// Returns (the trained model, every training batch's loss, validation losses).
fn train(steps: usize, dir: &Path) -> (UNet, Vec<f64>, Vec<f64>) {
    let device = Device::flex().autodiff();
    let config = TrainingConfig::new(
        UNetConfig::new(),
        AdamWConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(1e-4),
    )
    .with_steps(steps);
    std::fs::remove_dir_all(dir).ok();

    let train = DataLoaderBuilder::new(FlowBatcher)
        .batch_size(config.batch_size)
        .build(FlowDataset {
            seed: config.seed,
            len: config.steps * config.batch_size,
            p_drop: config.p_drop,
        });
    let valid = DataLoaderBuilder::new(FlowBatcher)
        .batch_size(config.batch_size)
        .build(FlowDataset {
            seed: 7, // a different stream: images, noise and times the model never trained on
            len: 8 * config.batch_size,
            p_drop: config.p_drop,
        });

    let training = SupervisedTraining::new(dir, train, valid)
        .metrics((LossMetric::new(),))
        .num_epochs(1) // one pass over steps x N fresh items
        .renderer(Box::new(Quiet));
    let model = config.model.init(&device).with_fixed_weights(&device);
    let learner = Learner::new(model, config.optimizer.init(), config.learning_rate);
    let result = training.launch(learner);
    if let Some(err) = result.error {
        panic!("training stopped early: {err}");
    }
    config.save(dir.join("config.json")).unwrap();
    (
        result.model,
        batch_losses(dir, "train"),
        batch_losses(dir, "valid"),
    )
}

fn main() {
    let dir = std::env::temp_dir().join("learning-burn-c18e2");
    let start = Instant::now();
    let steps = TrainingConfig::new(UNetConfig::new(), AdamWConfig::new()).steps;
    let (model, losses, valid) = train(steps, &dir);
    let secs = start.elapsed().as_secs_f64();

    println!("step range    mean train loss");
    for (k, chunk) in losses.chunks(200).enumerate() {
        let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
        println!(
            "{:>4}-{:<4}    {mean:.4}",
            k * 200 + 1,
            k * 200 + chunk.len()
        );
    }
    let v = valid.iter().sum::<f64>() / valid.len() as f64;
    println!("validation loss (8 batches, unseen stream) = {v:.4}");
    println!("{} steps of N = 64 in {secs:.1} s", losses.len());

    // Two files out: burnpack for Burn, safetensors (PyTorch names and layout) for Python.
    let out = weights_dir();
    std::fs::create_dir_all(&out).unwrap();
    model.clone().save_file(out.join("flow_unet")).unwrap(); // adds .bpk
    let mut store = SafetensorsStore::from_file(out.join("flow_unet.safetensors"))
        .overwrite(true)
        .with_to_adapter(BurnToPyTorchAdapter);
    model.save_into(&mut store).unwrap();
    std::fs::copy(
        dir.join("train/epoch-1/Loss.log"),
        out.join("flow_unet_loss.log"),
    )
    .unwrap();
    println!("saved weights/flow_unet.bpk and weights/flow_unet.safetensors");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // Five Learner steps from the fixed weights, against PyTorch's loop (python/main.py).
        let dir = std::env::temp_dir().join("learning-burn-c18e2-test");
        let (_, losses, _) = train(5, &dir);
        for (got, want) in losses.iter().zip(EXPECTED_FIRST_LOSSES) {
            assert!((got - want).abs() < 1e-3, "{got} vs {want}");
        }
        assert_eq!(losses.len(), 5);
    }

    const EXPECTED_FIRST_LOSSES: [f64; 5] = [2.915223, 3.726287, 2.406769, 1.856125, 1.609259];
}
