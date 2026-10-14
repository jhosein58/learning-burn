use burn::module::Param;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::activation::relu;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData};

// Chapter 23 -- A Stream wrapper for Conv1d, after Google's kws_streaming.
//
// One trained layer, four ways to run it. Training and non-streaming inference
// see the whole clip, padded on the left. The two streaming modes see s new
// frames per call and keep the last B = max(0, d(k-1) + 1 - s) frames, either
// inside the wrapper (internal state) or handed in and out by the caller
// (external state). All four must give the same numbers.

// The four modes defined by kws_streaming (Rybakov et al., 2020).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Training,
    NonStreamInference,
    StreamInternalState,
    StreamExternalState,
}

/// Frames a streaming conv must remember: its span d(k-1)+1, minus the s new ones.
fn buffer_len(k: usize, d: usize, s: usize) -> usize {
    (d * (k - 1) + 1).saturating_sub(s)
}

struct Stream {
    conv: Conv1d,             // the trained layer; left padding B for whole clips
    mode: Mode,               // what forward / step are allowed to do
    buffer: usize,            // B, in frames
    state: Option<Tensor<3>>, // [N, c_in, B], used by StreamInternalState only
}

impl Stream {
    fn new(mut conv: Conv1d, mode: Mode) -> Self {
        let buffer = buffer_len(conv.kernel_size, conv.dilation, conv.stride);
        assert!(buffer > 0, "B = 0: this layer needs no state");
        // Whole clips get the same B frames of history: zeros on the left.
        conv.padding = PaddingConfig1d::Explicit(buffer, 0);
        Self {
            conv,
            mode,
            buffer,
            state: None,
        }
    }

    /// The state a stream starts from: B frames of zeros, the causal padding.
    fn zero_state(&self, batch: usize, device: &Device) -> Tensor<3> {
        let c_in = self.conv.weight.dims()[1];
        Tensor::zeros([batch, c_in, self.buffer], device)
    }

    /// Training / NonStreamInference: the whole clip [N, c_in, T] at once.
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        assert!(matches!(
            self.mode,
            Mode::Training | Mode::NonStreamInference
        ));
        self.conv.forward(x)
    }

    /// StreamExternalState: s new frames [N, c_in, s] plus the state in,
    /// one output frame [N, c_out, 1] plus the new state out.
    fn step(&self, chunk: Tensor<3>, state: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        assert_eq!(self.mode, Mode::StreamExternalState);
        self.step_inner(chunk, state)
    }

    /// StreamInternalState: the same step, with the state kept in the wrapper.
    fn step_internal(&mut self, chunk: Tensor<3>) -> Tensor<3> {
        assert_eq!(self.mode, Mode::StreamInternalState);
        let [batch, _, _] = chunk.dims();
        let state = match self.state.take() {
            Some(s) => s,
            None => self.zero_state(batch, &chunk.device()),
        };
        let (y, state) = self.step_inner(chunk, state);
        self.state = Some(state);
        y
    }

    fn step_inner(&self, chunk: Tensor<3>, state: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        let window = Tensor::cat(vec![state, chunk], 2); // [N, c_in, B + s]
        let len = window.dims()[2];
        let options = ConvOptions::new([self.conv.stride], [0], [self.conv.dilation], 1);
        let bias = self.conv.bias.as_ref().map(|b| b.val());
        let y = conv1d(window.clone(), self.conv.weight.val(), bias, options);
        let state = window.narrow(2, len - self.buffer, self.buffer); // keep the newest B
        (y, state)
    }
}

// A tiny LCG so Rust and Python draw exactly the same numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    /// n values (next % (2m+1) - m) / div: exact in f32 when div is a power of 2.
    fn grid(&mut self, n: usize, m: u32, div: f32) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() % (2 * m + 1)) as f32 - m as f32) / div)
            .collect()
    }
}

/// A Conv1d with fixed weights from the LCG.
fn conv(c_in: usize, c_out: usize, k: usize, d: usize, s: usize, rng: &mut Lcg) -> Conv1d {
    let device = Device::flex();
    let mut conv = Conv1dConfig::new(c_in, c_out, k)
        .with_dilation(d)
        .with_stride(s)
        .init(&device);
    let w = rng.grid(c_out * c_in * k, 16, 32.0);
    let b = rng.grid(c_out, 16, 32.0);
    conv.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(w, [c_out, c_in, k]),
        &device,
    ));
    conv.bias = Some(Param::from_tensor(Tensor::from_data(
        TensorData::new(b, [c_out]),
        &device,
    )));
    conv
}

fn max_diff(a: Tensor<3>, b: Tensor<3>) -> f32 {
    (a - b).abs().max().into_scalar::<f32>()
}

const T: usize = 24;
const C_IN: usize = 2;
const C_OUT: usize = 3;
// (kernel, dilation, stride)
const CASES: [(usize, usize, usize); 5] = [(3, 1, 1), (3, 4, 1), (5, 1, 2), (3, 3, 2), (4, 2, 3)];

#[allow(dead_code)] // the fields are read by the test
struct Results {
    whole_first: Vec<f32>,  // case 0, channel 0, first 6 outputs
    diffs: Vec<(f32, f32)>, // per case: max|whole - internal|, max|whole - external|
    tcn_diff: f32,          // the stack: max|whole - external|
    tcn_first: Vec<f32>,    // the stack, channel 0, first 6 outputs
}

fn build() -> Results {
    let device = Device::flex();
    let mut rng = Lcg(2020);
    let x = Tensor::<3>::from_data(
        TensorData::new(rng.grid(C_IN * T, 8, 8.0), [1, C_IN, T]),
        &device,
    );

    println!(" k  d  s  B  whole        internal     external     max|diff| int  ext");
    let mut diffs = Vec::new();
    let mut whole_first = Vec::new();
    for (case, &(k, d, s)) in CASES.iter().enumerate() {
        let layer = conv(C_IN, C_OUT, k, d, s, &mut rng);

        // Training / non-streaming: the whole clip.
        let whole = Stream::new(layer.clone(), Mode::NonStreamInference).forward(x.clone());

        // Streaming, internal state: the wrapper remembers.
        let mut internal = Stream::new(layer.clone(), Mode::StreamInternalState);
        let ys: Vec<Tensor<3>> = (0..T / s)
            .map(|j| internal.step_internal(x.clone().narrow(2, j * s, s)))
            .collect();
        let y_int = Tensor::cat(ys, 2);

        // Streaming, external state: the caller carries the state.
        let external = Stream::new(layer, Mode::StreamExternalState);
        let mut state = external.zero_state(1, &device);
        let mut ys = Vec::new();
        for j in 0..T / s {
            let (y, next) = external.step(x.clone().narrow(2, j * s, s), state);
            ys.push(y);
            state = next;
        }
        let y_ext = Tensor::cat(ys, 2);

        let shape = |t: &Tensor<3>| format!("{:?}", t.dims());
        let (di, de) = (
            max_diff(whole.clone(), y_int.clone()),
            max_diff(whole.clone(), y_ext.clone()),
        );
        println!(
            "{k:>2} {d:>2} {s:>2} {:>2}  {:<12} {:<12} {:<12} {di:>12} {de:>4}",
            external.buffer,
            shape(&whole),
            shape(&y_int),
            shape(&y_ext)
        );
        if case == 0 {
            whole_first = whole
                .narrow(1, 0, 1)
                .narrow(2, 0, 6)
                .into_data()
                .try_to_vec()
                .unwrap();
        }
        diffs.push((di, de));
    }

    // A small TCN: three dilated causal convs with ReLU, streamed with external state.
    let stack: Vec<Stream> = [(C_IN, 4, 1), (4, 4, 2), (4, 4, 4)]
        .iter()
        .map(|&(ci, co, d)| conv(ci, co, 3, d, 1, &mut rng))
        .map(|c| Stream::new(c, Mode::Training)) // as trained: whole clips
        .collect();
    let mut whole = x.clone();
    for (i, layer) in stack.iter().enumerate() {
        whole = layer.forward(whole);
        if i + 1 < stack.len() {
            whole = relu(whole);
        }
    }
    let stack: Vec<Stream> = stack
        .into_iter()
        .map(|l| Stream::new(l.conv, Mode::StreamExternalState))
        .collect();
    let mut states: Vec<Tensor<3>> = stack.iter().map(|l| l.zero_state(1, &device)).collect();
    let mut ys = Vec::new();
    for t in 0..T {
        let mut h = x.clone().narrow(2, t, 1);
        for (i, layer) in stack.iter().enumerate() {
            let (y, next) = layer.step(h, states[i].clone());
            states[i] = next;
            h = if i + 1 < stack.len() { relu(y) } else { y };
        }
        ys.push(h);
    }
    let streamed = Tensor::cat(ys, 2);
    let state_frames: Vec<usize> = stack.iter().map(|l| l.buffer).collect();
    let state_floats: usize = states
        .iter()
        .map(|s| s.dims().iter().product::<usize>())
        .sum();
    let tcn_diff = max_diff(whole.clone(), streamed);
    println!("TCN: buffers {state_frames:?} frames, external state = {state_floats} floats");
    println!(
        "TCN: whole {:?}, max|whole - stream| = {tcn_diff}",
        whole.dims()
    );
    let tcn_first: Vec<f32> = whole
        .narrow(1, 0, 1)
        .narrow(2, 0, 6)
        .into_data()
        .try_to_vec()
        .unwrap();
    println!("TCN y[0][..6] = {tcn_first:?}");

    Results {
        whole_first,
        diffs,
        tcn_diff,
        tcn_first,
    }
}

fn main() {
    build();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let r = build();
        // main.py's Stream module (F.pad + F.conv1d, torch.cat buffers) prints these.
        let expected_whole = [
            0.171875, 0.6484375, -0.4296875, -0.1640625, 0.45703125, 0.66015625,
        ];
        let expected_tcn = [
            -0.50109863,
            -0.48828125,
            -0.55715942,
            -0.45449829,
            -0.70116425,
            -0.3212204,
        ];
        for (a, b) in r.whole_first.iter().zip(expected_whole) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        for (a, b) in r.tcn_first.iter().zip(expected_tcn) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        // Streaming equals non-streaming, in both state modes and for every case.
        for (di, de) in r.diffs {
            assert!(di < 1e-6 && de < 1e-6);
        }
        assert!(r.tcn_diff < 1e-6);
    }
}
