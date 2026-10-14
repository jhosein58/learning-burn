use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, DatasetError};
use burn::module::{Module, Param};
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::AdamWConfig;
use burn::optim::lr_scheduler::LrSchedulerConfig;
use burn::optim::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::optim::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::optim::lr_scheduler::sequential::SequentialLrSchedulerConfig;
use burn::tensor::{Device, Int, Tensor, TensorData};
use burn::train::logger::{EvaluationProgressLogger, TrainingProgressLogger};
use burn::train::metric::{
    AccuracyMetric, LearningRateMetric, LossMetric, MetricDefinition, PerplexityMetric,
};
use burn::train::renderer::{
    EvaluationName, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining,
};
use burn::train::{
    InferenceStep, Learner, SequenceOutput, SupervisedTraining, TrainOutput, TrainStep,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

// Chapter 16 -- Example 16.2's GPT, trained on Tiny Shakespeare by the Learner.

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/shakespeare.txt");

// --- the data: Example 16.1's tokenizer, windows and batcher ---
#[derive(Clone, Debug)]
struct CharTokenizer {
    chars: Vec<char>,
    ids: [Option<u32>; 128],
}

impl CharTokenizer {
    fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let mut ids = [None; 128];
        for (i, &c) in chars.iter().enumerate() {
            assert!(c.is_ascii(), "this tokenizer only handles ASCII");
            ids[c as usize] = Some(i as u32);
        }
        Self { chars, ids }
    }

    fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| self.ids[c as usize].expect("character not in the vocabulary"))
            .collect()
    }
}

#[derive(Clone, Debug)]
struct Window {
    tokens: Vec<u32>, // T + 1 tokens
}

struct TextWindows {
    tokens: Arc<Vec<u32>>,
    block: usize,
    stride: usize,
}

impl TextWindows {
    fn new(tokens: Vec<u32>, block: usize, stride: usize) -> Self {
        Self {
            tokens: Arc::new(tokens),
            block,
            stride,
        }
    }
}

impl Dataset<Window> for TextWindows {
    fn get(&self, index: usize) -> Result<Window, DatasetError> {
        assert!(index < self.len(), "window {index} out of range");
        let start = index * self.stride;
        Ok(Window {
            tokens: self.tokens[start..start + self.block + 1].to_vec(),
        })
    }

    fn len(&self) -> usize {
        (self.tokens.len() - self.block - 1) / self.stride + 1
    }
}

#[derive(Clone, Debug)]
struct TextBatch {
    inputs: Tensor<2, Int>,  // [N, T]
    targets: Tensor<2, Int>, // [N, T]
}

#[derive(Clone, Default)]
struct TextBatcher;

impl Batcher<Window, TextBatch> for TextBatcher {
    fn batch(&self, items: Vec<Window>, device: &Device) -> TextBatch {
        let n = items.len();
        let t = items[0].tokens.len() - 1;
        let mut inputs = Vec::with_capacity(n * t);
        let mut targets = Vec::with_capacity(n * t);
        for w in &items {
            inputs.extend(w.tokens[..t].iter().map(|&id| id as i64));
            targets.extend(w.tokens[1..].iter().map(|&id| id as i64));
        }
        TextBatch {
            inputs: Tensor::from_data(TensorData::new(inputs, [n, t]), device),
            targets: Tensor::from_data(TensorData::new(targets, [n, t]), device),
        }
    }
}

fn split(tokens: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let cut = tokens.len() * 9 / 10;
    (tokens[..cut].to_vec(), tokens[cut..].to_vec())
}

// --- the model: Example 16.2's GPT ---
#[derive(Config, Debug)]
struct GptConfig {
    vocab_size: usize,
    #[config(default = 128)]
    block_size: usize,
    #[config(default = 64)]
    d_model: usize,
    #[config(default = 4)]
    n_heads: usize,
    #[config(default = 2)]
    n_layers: usize,
    #[config(default = 0.0)]
    dropout: f64,
}

impl GptConfig {
    fn init(&self, device: &Device) -> Gpt {
        let d = self.d_model;
        Gpt {
            tok: EmbeddingConfig::new(self.vocab_size, d).init(device),
            pos: EmbeddingConfig::new(self.block_size, d).init(device),
            blocks: TransformerEncoderConfig::new(d, 4 * d, self.n_heads, self.n_layers)
                .with_norm_first(true)
                .with_dropout(self.dropout)
                .init(device),
            ln_f: LayerNormConfig::new(d).init(device),
            head: LinearConfig::new(d, self.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct Gpt {
    tok: Embedding,
    pos: Embedding,
    blocks: TransformerEncoder,
    ln_f: LayerNorm,
    head: Linear,
}

impl Gpt {
    fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let [n, t] = tokens.dims();
        let device = tokens.device();
        let positions = Tensor::<1, Int>::arange(0..t as i64, &device).reshape([1, t]);
        let x = self.tok.forward(tokens) + self.pos.forward(positions);
        let causal = generate_autoregressive_mask(n, t, &device);
        let x = self
            .blocks
            .forward(TransformerEncoderInput::new(x).mask_attn(causal));
        self.head.forward(self.ln_f.forward(x))
    }
}

// Example 16.2's fixed LCG weights, so the run starts where PyTorch's does.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF
    }
    fn uniform(&mut self, n: usize, bound: f64) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() as f64 / 32768.0 * 2.0 - 1.0) * bound) as f32)
            .collect()
    }
    fn embedding(&mut self, e: &mut Embedding, rows: usize, d: usize, device: &Device) {
        e.weight = param(self.uniform(rows * d, 0.1), [rows, d], device);
    }
    fn linear(&mut self, l: &mut Linear, d_in: usize, d_out: usize, device: &Device) {
        let bound = 1.0 / (d_in as f64).sqrt();
        let w = Tensor::<2>::from_data(
            TensorData::new(self.uniform(d_out * d_in, bound), [d_out, d_in]),
            device,
        );
        l.weight = Param::from_tensor(w.transpose());
        if l.bias.is_some() {
            l.bias = Some(param(self.uniform(d_out, bound), [d_out], device));
        }
    }
    fn norm(&mut self, ln: &mut LayerNorm, d: usize, device: &Device) {
        let g: Vec<f32> = self.uniform(d, 0.1).iter().map(|v| 1.0 + v).collect();
        ln.gamma = param(g, [d], device);
        ln.beta = Some(param(self.uniform(d, 0.1), [d], device));
    }
}

fn param<const D: usize>(v: Vec<f32>, shape: [usize; D], device: &Device) -> Param<Tensor<D>> {
    Param::from_tensor(Tensor::from_data(TensorData::new(v, shape), device))
}

impl Gpt {
    fn with_fixed_weights(mut self, seed: u32, device: &Device) -> Self {
        let [v, d] = self.tok.weight.dims();
        let t = self.pos.weight.dims()[0];
        let mut f = Lcg(seed);
        f.embedding(&mut self.tok, v, d, device);
        f.embedding(&mut self.pos, t, d, device);
        for layer in self.blocks.layers.iter_mut() {
            f.norm(&mut layer.norm_2, d, device);
            f.linear(&mut layer.mha.query, d, d, device);
            f.linear(&mut layer.mha.key, d, d, device);
            f.linear(&mut layer.mha.value, d, d, device);
            f.linear(&mut layer.mha.output, d, d, device);
            f.norm(&mut layer.norm_1, d, device);
            f.linear(&mut layer.pwff.linear_inner, d, 4 * d, device);
            f.linear(&mut layer.pwff.linear_outer, 4 * d, d, device);
        }
        f.norm(&mut self.ln_f, d, device);
        f.linear(&mut self.head, d, v, device);
        self
    }
}

// --- the two traits the Learner needs ---
impl Gpt {
    fn forward_sequence(&self, batch: TextBatch) -> SequenceOutput {
        let logits = self.forward(batch.inputs); // [N, T, V]
        let [n, t, v] = logits.dims();
        // Every one of the N x T positions is a prediction: one cross-entropy over them all.
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(
                logits.clone().reshape([n * t, v]),
                batch.targets.clone().reshape([n * t]),
            );
        SequenceOutput::new(loss, logits, None, batch.targets)
    }
}

impl TrainStep for Gpt {
    type Input = TextBatch;
    type Output = SequenceOutput;

    fn step(&self, batch: TextBatch) -> TrainOutput<SequenceOutput> {
        let item = self.forward_sequence(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for Gpt {
    type Input = TextBatch;
    type Output = SequenceOutput;

    fn step(&self, batch: TextBatch) -> SequenceOutput {
        self.forward_sequence(batch)
    }
}

// --- the run, written down ---
#[derive(Config, Debug)]
struct TrainingConfig {
    model: GptConfig,
    optimizer: AdamWConfig,
    #[config(default = 8)]
    num_epochs: usize,
    #[config(default = 16)]
    batch_size: usize,
    #[config(default = 1e-2)]
    learning_rate: f64, // the peak
    #[config(default = 1e-3)]
    min_learning_rate: f64,
    #[config(default = 50)]
    warmup_steps: usize,
    #[config(default = 16)]
    seed: u64,
}

// Linear warm-up to the peak, then a cosine to the minimum, over `total` steps.
fn schedule(c: &TrainingConfig, total: usize) -> SequentialLrSchedulerConfig {
    let w = c.warmup_steps;
    let peak = c.learning_rate;
    SequentialLrSchedulerConfig::new(
        vec![
            LrSchedulerConfig::Linear(LinearLrSchedulerConfig::new(peak / w as f64, peak, w - 1)),
            LrSchedulerConfig::Cosine(
                CosineAnnealingLrSchedulerConfig::new(peak, total - w)
                    .with_min_lr(c.min_learning_rate),
            ),
        ],
        vec![w],
    )
}

fn training_config(vocab: usize) -> TrainingConfig {
    TrainingConfig::new(
        GptConfig::new(vocab),
        AdamWConfig::new()
            .with_epsilon(1e-8) // PyTorch's default; Burn's is 1e-5
            .with_weight_decay(0.01), // PyTorch's default; Burn's is 1e-4
    )
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

// The batch lines of <dir>/<split>/epoch-<e>/<metric>.log ("value,count"), and the
// Learner's own "value,final" line for the epoch.
fn metric_log(dir: &Path, split: &str, epoch: usize, metric: &str) -> (Vec<(f64, f64)>, f64) {
    let log = std::fs::read_to_string(dir.join(format!("{split}/epoch-{epoch}/{metric}.log")))
        .expect("metric log");
    let (mut lines, mut last) = (Vec::new(), f64::NAN);
    for line in log.lines() {
        let (v, c) = line.split_once(',').unwrap();
        let v: f64 = v.parse().unwrap();
        match c {
            "final" => last = v,
            c => lines.push((v, c.parse().unwrap())),
        }
    }
    (lines, last)
}

/// The training program: train, write config.json, vocab.txt and model.bpk. Returns the per-epoch
/// (train loss, valid loss, valid perplexity, valid accuracy %) and the checkpoint files.
fn train(dir: &Path) -> (Vec<(f64, f64, f64, f64)>, Vec<String>) {
    let device = Device::flex().autodiff();
    let text = std::fs::read_to_string(DATA).expect("read shakespeare.txt");
    let tok = CharTokenizer::new(&text);
    let (train_ids, valid_ids) = split(&tok.encode(&text));
    let config = training_config(tok.vocab_size());
    let t = config.model.block_size;
    let train = TextWindows::new(train_ids, t, t);
    let valid = TextWindows::new(valid_ids, t, t);
    let steps = config.num_epochs * train.len().div_ceil(config.batch_size);
    std::fs::remove_dir_all(dir).ok();
    std::fs::create_dir_all(dir).unwrap();

    let train = DataLoaderBuilder::new(TextBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed) // a fresh order of the windows every epoch
        .build(train);
    let valid = DataLoaderBuilder::new(TextBatcher)
        .batch_size(config.batch_size)
        .build(valid);

    let training = SupervisedTraining::new(dir, train, valid)
        .metrics((
            LossMetric::new(),
            PerplexityMetric::new(),
            AccuracyMetric::new(),
        ))
        .metric_train_numeric(LearningRateMetric::new())
        .with_default_checkpointers()
        .num_epochs(config.num_epochs)
        .renderer(Box::new(Quiet));

    let model = config.model.init(&device).with_fixed_weights(42, &device);
    let lr = schedule(&config, steps).init().expect("valid schedule");
    let result = training.launch(Learner::new(model, config.optimizer.init(), lr));
    if let Some(err) = result.error {
        panic!("training stopped early: {err}");
    }

    // The whole deliverable: the config, the vocabulary and the weights.
    config.save(dir.join("config.json")).unwrap();
    std::fs::write(dir.join("vocab.txt"), tok.chars.iter().collect::<String>()).unwrap();
    result.model.save_file(dir.join("model")).unwrap(); // burnpack: adds .bpk

    let mean = |lines: Vec<(f64, f64)>| {
        let n: f64 = lines.iter().map(|l| l.1).sum();
        lines.iter().map(|l| l.0 * l.1).sum::<f64>() / n
    };
    let history = (1..=config.num_epochs)
        .map(|e| {
            (
                mean(metric_log(dir, "train", e, "Loss").0),
                mean(metric_log(dir, "valid", e, "Loss").0),
                metric_log(dir, "valid", e, "Perplexity").1,
                metric_log(dir, "valid", e, "Accuracy").1,
            )
        })
        .collect();
    let mut files: Vec<String> = std::fs::read_dir(dir.join("checkpoint"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    (history, files)
}

fn main() {
    let dir = std::env::temp_dir().join("learning-burn-c16");
    let start = Instant::now();
    let (history, files) = train(&dir);
    println!("epoch  train loss  valid loss  valid ppl  valid acc");
    for (e, (tl, vl, ppl, acc)) in history.iter().enumerate() {
        println!(
            "{:>5}  {tl:>10.4}  {vl:>10.4}  {ppl:>9.2}  {acc:>8.1}%",
            e + 1
        );
    }
    println!("checkpoint/ = {files:?}");
    for f in ["config.json", "vocab.txt", "model.bpk"] {
        let bytes = std::fs::metadata(dir.join(f)).unwrap().len();
        println!("{f:<12} {bytes:>7} bytes");
    }
    println!("trained in {:.0} s", start.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // The run itself shuffles with Burn's generator, so it is not compared number for
        // number. What is deterministic is checked: the step count and the whole schedule,
        // against python/main.py's schedule() at the steps where it changes character.
        let config = training_config(62);
        let steps = config.num_epochs * 1406usize.div_ceil(config.batch_size);
        assert_eq!(steps, 704);
        let mut lr = schedule(&config, steps).init().unwrap();
        let lrs: Vec<f64> = (0..steps).map(|_| lr.step().base()).collect();
        let expected = [
            (1, 0.0002),
            (49, 0.0098),
            (50, 0.01),
            (51, 0.01),
            (52, 0.0099999480810463),
            (100, 0.009875916828731804),
            (352, 0.006060568023042318),
            (704, 0.0010000519189537003),
        ];
        for (k, e) in expected {
            assert!((lrs[k - 1] - e).abs() < 1e-12, "step {k}");
        }
        let device = Device::flex();
        assert_eq!(config.model.init(&device).num_params(), 116224);
    }
}
