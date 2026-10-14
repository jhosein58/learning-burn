use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, DatasetError};
use burn::module::Module;
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
use std::path::{Path, PathBuf};

const SIZE: usize = 12;
const PIXELS: usize = SIZE * SIZE;
const NULL_CLASS: usize = 4; // the "no label" class used for classifier-free guidance

// A tiny LCG so Rust and Python draw exactly the same "random" numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
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

// One training example of flow matching: data x1, noise x0, a time t, a label.
#[derive(Clone, Debug)]
struct FlowItem {
    x1: Vec<f32>,
    x0: Vec<f32>,
    t: f32,
    label: usize,
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

// --- sampling: Euler steps along the learned velocity, with classifier-free guidance ---
// x_{k+1} = x_k + dt * [ v_u + w (v_c - v_u) ],  t_k = k dt,  dt = 1/K   (from t = 0 to t = 1)
fn sample(model: &UNet, noise: Tensor<4>, labels: &[usize], w: f32, steps: usize) -> Tensor<4> {
    let device = noise.device();
    let n = labels.len();
    // One forward pass per step does both: the first n rows conditioned, the last n not.
    let y: Vec<i64> = labels
        .iter()
        .map(|&c| c as i64)
        .chain(std::iter::repeat_n(NULL_CLASS as i64, n))
        .collect();
    let y = Tensor::<1, Int>::from_data(TensorData::new(y, [2 * n]), &device);
    let dt = 1.0 / steps as f32;
    let mut x = noise;
    for k in 0..steps {
        let t = Tensor::<1>::full([2 * n], k as f32 * dt, &device);
        let v = model.forward(Tensor::cat(vec![x.clone(), x.clone()], 0), t, y.clone());
        let v_c = v.clone().slice(0..n);
        let v_u = v.slice(n..2 * n);
        x = x + (v_u.clone() + (v_c - v_u).mul_scalar(w)).mul_scalar(dt);
    }
    x
}

// --- judging a sample: the nearest clean shape, over every class and position ---
// A clean template is 0.8 on the shape and -0.8 elsewhere: the mean of the data's two levels.
fn templates() -> Vec<(usize, Vec<f32>)> {
    let mut out = Vec::new();
    let mut add = |class: usize, cells: Vec<usize>| {
        let mut img = vec![-0.8f32; PIXELS];
        cells.into_iter().for_each(|p| img[p] = 0.8);
        out.push((class, img));
    };
    for r in 0..12 {
        for c in 0..7 {
            add(0, (0..6).map(|j| r * SIZE + c + j).collect());
        }
    }
    for c in 0..12 {
        for r in 0..7 {
            add(1, (0..6).map(|i| (r + i) * SIZE + c).collect());
        }
    }
    for r in 0..7 {
        for c in 0..7 {
            add(2, (0..6).map(|i| (r + i) * SIZE + c + i).collect());
        }
    }
    for r in 2..10 {
        for c in 2..10 {
            let plus = (0..5).flat_map(|d| [r * SIZE + c + d - 2, (r + d - 2) * SIZE + c]);
            add(3, plus.collect());
        }
    }
    out
}

/// For each image: (class of the nearest template, mean squared distance to it).
fn judge(images: &[f32], templates: &[(usize, Vec<f32>)]) -> Vec<(usize, f32)> {
    images
        .chunks(PIXELS)
        .map(|img| {
            templates
                .iter()
                .map(|(class, t)| {
                    let d = img
                        .iter()
                        .zip(t)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>();
                    (*class, d / PIXELS as f32)
                })
                .fold(
                    (0, f32::INFINITY),
                    |best, x| if x.1 < best.1 { x } else { best },
                )
        })
        .collect()
}

/// (percent of samples whose nearest shape is the requested class, mean distance).
fn score(images: &[f32], labels: &[usize], templates: &[(usize, Vec<f32>)]) -> (f32, f32) {
    let judged = judge(images, templates);
    let hits = judged
        .iter()
        .zip(labels)
        .filter(|((c, _), l)| c == *l)
        .count();
    let dist = judged.iter().map(|(_, d)| d).sum::<f32>() / judged.len() as f32;
    (100.0 * hits as f32 / labels.len() as f32, dist)
}

// A grid of 12x12 tiles, `cols` per row, 1-pixel grey gaps, as an 8-bit PNG.
fn save_grid(images: &[f32], cols: usize, path: &Path) {
    let rows = images.len() / PIXELS / cols;
    let (w, h) = (cols * (SIZE + 1) + 1, rows * (SIZE + 1) + 1);
    let mut buf = vec![128u8; w * h];
    for (k, img) in images.chunks(PIXELS).enumerate() {
        let (r0, c0) = ((k / cols) * (SIZE + 1) + 1, (k % cols) * (SIZE + 1) + 1);
        for (p, &x) in img.iter().enumerate() {
            let v = ((x + 1.0) / 2.0).clamp(0.0, 1.0); // [-1, 1] -> [0, 1]
            buf[(r0 + p / SIZE) * w + c0 + p % SIZE] = (v * 255.0).round() as u8;
        }
    }
    image::GrayImage::from_raw(w as u32, h as u32, buf)
        .unwrap()
        .save(path)
        .unwrap();
}

fn noise(n: usize, seed: u32, device: &Device) -> Tensor<4> {
    let z = Lcg(seed).normals(n * PIXELS);
    Tensor::from_data(TensorData::new(z, [n, 1, SIZE, SIZE]), device)
}

fn weights_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights")
}

fn vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec().unwrap()
}

fn trained(device: &Device) -> UNet {
    UNet::new(16, 32, 64, device).load_file(weights_dir().join("flow_unet"))
}

const W: f32 = 2.0; // the guidance scale of the teacher
const TEACHER_STEPS: usize = 16;

// --- reflow: pair each noise with the sample the model makes from it ---
// x0 -> x1 = sample(x0): a coupling, not two independent draws. A straight path between
// the two ends of each pair is what the second model learns.
fn make_pairs(model: &UNet, m: usize, seed: u32) -> Vec<(Vec<f32>, Vec<f32>, usize)> {
    let device = Device::flex();
    let labels: Vec<usize> = (0..m).map(|i| i % 4).collect();
    let x0 = vec(noise(m, seed, &device));
    let x1 = vec(sample(
        model,
        noise(m, seed, &device),
        &labels,
        W,
        TEACHER_STEPS,
    ));
    (0..m)
        .map(|i| {
            let px = i * PIXELS..(i + 1) * PIXELS;
            (x0[px.clone()].to_vec(), x1[px].to_vec(), labels[i])
        })
        .collect()
}

// The pairs again and again, each time with a fresh t: item i is pair i mod m.
struct ReflowDataset {
    pairs: Vec<(Vec<f32>, Vec<f32>, usize)>,
    len: usize,
}

impl Dataset<FlowItem> for ReflowDataset {
    fn get(&self, index: usize) -> Result<FlowItem, DatasetError> {
        assert!(index < self.len, "index {index} out of range");
        let (x0, x1, label) = &self.pairs[index % self.pairs.len()];
        let mut rng = Lcg(31 ^ (index as u32).wrapping_mul(2654435761));
        Ok(FlowItem {
            x1: x1.clone(),
            x0: x0.clone(),
            t: rng.uniform(),
            label: *label,
        })
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Fine-tunes `model` on the pairs with the Learner. Returns (model, every batch's loss).
fn reflow(
    model: UNet,
    pairs: Vec<(Vec<f32>, Vec<f32>, usize)>,
    steps: usize,
    dir: &Path,
) -> (UNet, Vec<f64>) {
    std::fs::remove_dir_all(dir).ok();
    let valid_pairs = pairs[..64].to_vec();
    let train = DataLoaderBuilder::new(FlowBatcher)
        .batch_size(64)
        .build(ReflowDataset {
            pairs,
            len: steps * 64,
        });
    let valid = DataLoaderBuilder::new(FlowBatcher)
        .batch_size(64)
        .build(ReflowDataset {
            pairs: valid_pairs,
            len: 64,
        });
    let training = SupervisedTraining::new(dir, train, valid)
        .metrics((LossMetric::new(),))
        .num_epochs(1)
        .renderer(Box::new(Quiet));
    let optim = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(1e-4);
    let result = training.launch(Learner::new(model, optim.init(), 1e-3));
    if let Some(err) = result.error {
        panic!("training stopped early: {err}");
    }
    (result.model, batch_losses(dir, "train"))
}

/// Mean squared distance per pixel between two batches of images.
fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32
}

struct Row {
    model: &'static str,
    w: f32,
    steps: usize,
    hits: f32,
    to_shape: f32,
    to_teacher: f32,
}

fn main() {
    let device = Device::flex();
    let base = trained(&device);
    let tpl = templates();
    let labels: Vec<usize> = (0..256).map(|i| i % 4).collect();
    let z = noise(256, 5, &device); // Example 18.3's noise: never used for training
    let teacher = vec(sample(&base, z.clone(), &labels, W, TEACHER_STEPS));
    let out = std::env::temp_dir().join("learning-burn-c18e4");
    std::fs::create_dir_all(&out).unwrap();

    let mut rows = Vec::new();
    let mut eval = |name, model: &UNet, w, steps| {
        let x = vec(sample(model, z.clone(), &labels, w, steps));
        let (hits, to_shape) = score(&x, &labels, &tpl);
        let to_teacher = distance(&x, &teacher);
        if steps == 1 {
            let grid: Vec<f32> = (0..4)
                .flat_map(|c| (0..8).map(move |j| (4 * j + c) * PIXELS))
                .flat_map(|s| x[s..s + PIXELS].to_vec())
                .collect();
            save_grid(&grid, 8, &out.join(format!("c18e4_{name}_1step.png")));
        }
        rows.push(Row {
            model: name,
            w,
            steps,
            hits,
            to_shape,
            to_teacher,
        });
    };
    for steps in [1, 2, 4, 8, 16, 32] {
        eval("base", &base, W, steps);
    }

    // One round of reflow: 1024 pairs from the 16-step guided sampler, 400 steps of fine-tuning.
    let pairs = make_pairs(&base, 1024, 11);
    let dir = std::env::temp_dir().join("learning-burn-c18e4");
    let (student, losses) = reflow(
        trained(&Device::flex().autodiff()),
        pairs,
        400,
        &dir.join("run"),
    );
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    // The student learned the GUIDED sampler's paths, so it is sampled with w = 1: no guidance.
    for steps in [1, 2, 4] {
        eval("reflow", &student, 1.0, steps);
    }

    println!("model    w  steps  right shape  to nearest shape  to 16-step teacher");
    for r in &rows {
        println!(
            "{:<6} {:>3.1} {:>6} {:>11.1}% {:>17.4} {:>19.4}",
            r.model, r.w, r.steps, r.hits, r.to_shape, r.to_teacher
        );
    }
    println!(
        "reflow loss: first 50 steps {:.4}, last 50 steps {:.4}",
        mean(&losses[..50]),
        mean(&losses[losses.len() - 50..])
    );
    student
        .save_file(weights_dir().join("flow_unet_reflow"))
        .unwrap();
    std::fs::copy(
        dir.join("run/train/epoch-1/Loss.log"),
        out.join("reflow_loss.log"),
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // 64 pairs, then three reflow steps on them: python/main.py does the same.
        let pairs = make_pairs(&trained(&Device::flex()), 64, 11);
        for (i, want) in EXPECTED_PAIR.iter().enumerate() {
            assert!((pairs[3].1[i] - want).abs() < 1e-3, "pair pixel {i}");
        }
        let dir = std::env::temp_dir().join("learning-burn-c18e4-test");
        let (_, losses) = reflow(trained(&Device::flex().autodiff()), pairs, 3, &dir);
        for (got, want) in losses.iter().zip(EXPECTED_LOSSES) {
            assert!((got - want).abs() < 1e-3, "{got} vs {want}");
        }
    }

    const EXPECTED_PAIR: [f32; 4] = [-0.793831, -0.681093, -0.665306, -0.655993];
    const EXPECTED_LOSSES: [f64; 3] = [0.094534, 0.085923, 0.086509];
}
