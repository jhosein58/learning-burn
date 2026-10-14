use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::activation::relu;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Int, Tensor, TensorData};

// Chapter 23 -- Train on whole clips, run one frame at a time.
//
// A burst detector: three dilated causal convs, a GRU and a linear head, trained
// on whole clips of a noisy signal to say, frame by frame, "a sine burst is on".
// Then the same weights run one frame per call with an external state, and the
// two must agree.

const T: usize = 96; //      frames per clip
const BATCH: usize = 32; //  clips per step
const TRAIN: usize = 256; // training clips
const TEST: usize = 64; //   test clips
const K: usize = 3;
const DILATIONS: [usize; 3] = [1, 2, 4];
const C: usize = 8; //  conv channels
const H: usize = 16; // GRU hidden size

// A tiny LCG so Rust and Python draw exactly the same numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
    /// n values (next % (2m+1) - m) / div: exact in f32 when div is a power of 2.
    fn grid(&mut self, n: usize, m: u32, div: f32) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() % (2 * m + 1)) as f32 - m as f32) / div)
            .collect()
    }
}

// --- the data: noise everywhere, a sine burst somewhere, a label per frame ---
fn clip(rng: &mut Lcg) -> (Vec<f32>, Vec<i32>) {
    let len = 16 + rng.below(17); //            burst length 16..=32 frames
    let start = 8 + rng.below(T - 8 - len); //  burst start
    let mut x = rng.grid(T, 32, 64.0); //       uniform noise in [-0.5, 0.5]
    let mut y = vec![0; T];
    for t in start..start + len {
        let phase = 2.0 * std::f64::consts::PI * (t - start) as f64 / 8.0;
        x[t] += phase.sin() as f32; // period 8 frames, amplitude 1
        y[t] = 1;
    }
    (x, y)
}

fn dataset(n: usize, seed: u32) -> (Vec<f32>, Vec<i32>) {
    let mut rng = Lcg(seed);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let (x, y) = clip(&mut rng);
        xs.extend(x);
        ys.extend(y);
    }
    (xs, ys)
}

// --- the model ---
#[derive(Module, Debug)]
struct Detector {
    convs: Vec<Conv1d>, // causal: left padding d (k - 1)
    gru: Gru,
    head: Linear,
}

/// What a stream must carry from one frame to the next.
struct State {
    convs: Vec<Tensor<3>>, // [N, c_in, d (k - 1)] per conv
    h: Tensor<2>,          // [N, H]
}

impl Detector {
    fn new(device: &Device) -> Self {
        // Fixed weights from the LCG (main.py draws the same ones), zero biases.
        let mut rng = Lcg(7);
        let mut convs = Vec::new();
        for (i, &d) in DILATIONS.iter().enumerate() {
            let c_in = if i == 0 { 1 } else { C };
            let mut conv = Conv1dConfig::new(c_in, C, K)
                .with_dilation(d)
                .with_padding(PaddingConfig1d::Explicit(d * (K - 1), 0))
                .init(device);
            let div = if i == 0 { 32.0 } else { 128.0 }; // about 1/sqrt(fan_in)
            conv.weight = param(rng.grid(C * c_in * K, 16, div), [C, c_in, K], device);
            conv.bias = Some(param(vec![0.0; C], [C], device));
            convs.push(conv);
        }
        // Burn's Linear (inside every gate) stores [in, out].
        let mut gru = GruConfig::new(C, H, true).init(device);
        for g in [&mut gru.reset_gate, &mut gru.update_gate, &mut gru.new_gate] {
            g.input_transform.weight = param(rng.grid(C * H, 16, 64.0), [C, H], device);
            g.input_transform.bias = Some(param(vec![0.0; H], [H], device));
            g.hidden_transform.weight = param(rng.grid(H * H, 16, 64.0), [H, H], device);
            g.hidden_transform.bias = Some(param(vec![0.0; H], [H], device));
        }
        let mut head = LinearConfig::new(H, 1).init(device);
        head.weight = param(rng.grid(H, 16, 64.0), [H, 1], device);
        head.bias = Some(param(vec![0.0], [1], device));
        Self { convs, gru, head }
    }

    /// Whole clips: [N, 1, T] -> logits [N, T].
    fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        let mut h = x;
        for conv in &self.convs {
            h = relu(conv.forward(h)); //                    [N, C, T]
        }
        let h = self.gru.forward(h.swap_dims(1, 2), None); // [N, T, H]
        let [n, t, _] = h.dims();
        self.head.forward(h).reshape([n, t])
    }

    fn zero_state(&self, n: usize, device: &Device) -> State {
        let convs = self
            .convs
            .iter()
            .map(|c| Tensor::zeros([n, c.weight.dims()[1], c.dilation * (K - 1)], device))
            .collect();
        State {
            convs,
            h: Tensor::zeros([n, H], device),
        }
    }

    /// One frame: [N, 1, 1] and the state in, logits [N, 1] and the state out.
    fn step(&self, x_t: Tensor<3>, state: State) -> (Tensor<2>, State) {
        let mut h = x_t;
        let mut buffers = Vec::new();
        for (conv, buf) in self.convs.iter().zip(state.convs) {
            let window = Tensor::cat(vec![buf, h], 2); // [N, c_in, d (k - 1) + 1]
            let keep = window.dims()[2] - 1;
            let options = ConvOptions::new([1], [0], [conv.dilation], 1);
            let bias = conv.bias.as_ref().map(|b| b.val());
            h = relu(conv1d(window.clone(), conv.weight.val(), bias, options)); // [N, C, 1]
            buffers.push(window.narrow(2, 1, keep)); // drop the oldest frame
        }
        let [n, _, _] = h.dims();
        let out = self.gru.forward(h.swap_dims(1, 2), Some(state.h)); // [N, 1, H]
        let h_new = out.reshape([n, H]);
        let logit = self.head.forward(h_new.clone()); // [N, 1]
        (
            logit,
            State {
                convs: buffers,
                h: h_new,
            },
        )
    }
}

fn param<const D: usize>(v: Vec<f32>, shape: [usize; D], device: &Device) -> Param<Tensor<D>> {
    Param::from_tensor(Tensor::from_data(TensorData::new(v, shape), device))
}

fn batch(
    xs: &[f32],
    ys: &[i32],
    from: usize,
    n: usize,
    device: &Device,
) -> (Tensor<3>, Tensor<2, Int>) {
    let x = xs[from * T..(from + n) * T].to_vec();
    let y = ys[from * T..(from + n) * T].to_vec();
    (
        Tensor::from_data(TensorData::new(x, [n, 1, T]), device),
        Tensor::from_data(TensorData::new(y, [n, T]), device),
    )
}

/// Train for `epochs` passes over the training set; returns (model, loss per step).
fn train(epochs: usize, max_steps: usize) -> (Detector, Vec<f32>) {
    let device = Device::flex().autodiff();
    let (xs, ys) = dataset(TRAIN, 42);
    let mut model = Detector::new(&device);
    // Burn's Adam defaults to epsilon 1e-5; PyTorch's to 1e-8. Match PyTorch.
    let mut optim = AdamConfig::new().with_epsilon(1e-8).init();
    let loss_fn = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&device);
    let mut losses = Vec::new();
    'outer: for _ in 0..epochs {
        for b in 0..TRAIN / BATCH {
            let (x, y) = batch(&xs, &ys, b * BATCH, BATCH, &device);
            let loss = loss_fn.forward(model.forward(x), y);
            losses.push(loss.clone().into_scalar::<f32>());
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(0.01, model, grads);
            if losses.len() == max_steps {
                break 'outer;
            }
        }
    }
    (model, losses)
}

/// Per-frame logits for every test clip, whole-clip and streamed.
fn whole_and_streamed(model: &Detector, x: Tensor<3>) -> (Tensor<2>, Tensor<2>) {
    let device = x.device();
    let [n, _, t_len] = x.dims();
    let whole = model.forward(x.clone());
    let mut state = model.zero_state(n, &device);
    let mut frames = Vec::new();
    for t in 0..t_len {
        let (logit, next) = model.step(x.clone().narrow(2, t, 1), state);
        frames.push(logit);
        state = next;
    }
    (whole, Tensor::cat(frames, 1))
}

fn main() {
    let (model, losses) = train(12, usize::MAX);
    let steps = losses.len();
    for (i, l) in losses.iter().enumerate() {
        if i == 0 || (i + 1) % 24 == 0 {
            println!("step {:>3}  loss {l:.4}", i + 1);
        }
    }
    println!("{steps} steps of Adam, batch {BATCH}");

    // Inference: a validation snapshot, on a plain device.
    let model = model.valid();
    let device = Device::flex();
    let (xs, ys) = dataset(TEST, 7);
    let (x, _) = batch(&xs, &ys, 0, TEST, &device);
    let (whole, streamed) = whole_and_streamed(&model, x);
    let diff = (whole.clone() - streamed.clone())
        .abs()
        .max()
        .into_scalar::<f32>();
    let logits: Vec<f32> = streamed.into_data().try_to_vec().unwrap();

    // Frame accuracy, and how many frames after a burst starts the detector fires.
    let correct = logits
        .iter()
        .zip(&ys)
        .filter(|(z, y)| (**z > 0.0) == (**y == 1))
        .count();
    let mut delays = Vec::new();
    for c in 0..TEST {
        let y = &ys[c * T..(c + 1) * T];
        let z = &logits[c * T..(c + 1) * T];
        let start = y.iter().position(|&v| v == 1).unwrap();
        if let Some(d) = (start..T).position(|t| z[t] > 0.0) {
            delays.push(d as f32);
        }
    }
    let mean_delay = delays.iter().sum::<f32>() / delays.len() as f32;
    println!(
        "test frames correct = {correct}/{} ({:.1}%)",
        TEST * T,
        100.0 * correct as f32 / (TEST * T) as f32
    );
    println!(
        "bursts detected = {}/{TEST}, mean delay after onset = {mean_delay:.2} frames",
        delays.len()
    );
    println!("whole clip vs one frame per call, {TEST} clips x {T} frames:");
    println!("max|whole - streamed| = {diff:.1e}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // Loss of the first five Adam steps from main.py: same data, same init.
        let (model, losses) = train(1, 5);
        let expected = [0.692895, 0.680786, 0.665949, 0.645024, 0.623308];
        for (a, b) in losses.iter().zip(expected) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // And streaming equals whole-clip inference, even after training steps.
        let model = model.valid();
        let device = Device::flex();
        let (xs, ys) = dataset(8, 7);
        let (x, _) = batch(&xs, &ys, 0, 8, &device);
        let (whole, streamed) = whole_and_streamed(&model, x);
        let diff = (whole - streamed).abs().max().into_scalar::<f32>();
        assert!(diff < 1e-5, "{diff}");
    }
}
