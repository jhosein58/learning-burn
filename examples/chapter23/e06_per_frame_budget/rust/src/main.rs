use std::time::Instant;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::activation::relu;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData};

// Chapter 23 -- What one frame costs.
//
// The detector of Example 23.5, priced per frame: multiply-adds, parameters and
// state by formula, then timed on this laptop's CPU four ways -- whole clip,
// recompute the window every frame, stream with Burn, and stream with plain
// Rust slices (the shape firmware takes). All four must give the same logits.

const T: usize = 96;
const K: usize = 3;
const DILATIONS: [usize; 3] = [1, 2, 4];
const C: usize = 8;
const H: usize = 16;
const CLIPS: usize = 20; // clips streamed for the timing

// A tiny LCG so Rust and Python draw exactly the same numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    fn grid(&mut self, n: usize, m: u32, div: f32) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() % (2 * m + 1)) as f32 - m as f32) / div)
            .collect()
    }
}

fn param<const D: usize>(v: Vec<f32>, shape: [usize; D], device: &Device) -> Param<Tensor<D>> {
    Param::from_tensor(Tensor::from_data(TensorData::new(v, shape), device))
}

fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec().unwrap()
}

// --- the model of Example 23.5, with its initial (LCG) weights ---
#[derive(Module, Debug)]
struct Detector {
    convs: Vec<Conv1d>,
    gru: Gru,
    head: Linear,
}

struct State {
    convs: Vec<Tensor<3>>,
    h: Tensor<2>,
}

impl Detector {
    fn new(device: &Device) -> Self {
        let mut rng = Lcg(7);
        let mut convs = Vec::new();
        for (i, &d) in DILATIONS.iter().enumerate() {
            let c_in = if i == 0 { 1 } else { C };
            let mut conv = Conv1dConfig::new(c_in, C, K)
                .with_dilation(d)
                .with_padding(PaddingConfig1d::Explicit(d * (K - 1), 0))
                .init(device);
            let div = if i == 0 { 32.0 } else { 128.0 };
            conv.weight = param(rng.grid(C * c_in * K, 16, div), [C, c_in, K], device);
            conv.bias = Some(param(vec![0.0; C], [C], device));
            convs.push(conv);
        }
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

    fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        let mut h = x;
        for conv in &self.convs {
            h = relu(conv.forward(h));
        }
        let h = self.gru.forward(h.swap_dims(1, 2), None);
        let [n, t, _] = h.dims();
        self.head.forward(h).reshape([n, t])
    }

    fn zero_state(&self, device: &Device) -> State {
        let convs = self
            .convs
            .iter()
            .map(|c| Tensor::zeros([1, c.weight.dims()[1], c.dilation * (K - 1)], device))
            .collect();
        State {
            convs,
            h: Tensor::zeros([1, H], device),
        }
    }

    fn step(&self, x_t: Tensor<3>, state: State) -> (Tensor<2>, State) {
        let mut h = x_t;
        let mut buffers = Vec::new();
        for (conv, buf) in self.convs.iter().zip(state.convs) {
            let window = Tensor::cat(vec![buf, h], 2);
            let keep = window.dims()[2] - 1;
            let options = ConvOptions::new([1], [0], [conv.dilation], 1);
            let bias = conv.bias.as_ref().map(|b| b.val());
            h = relu(conv1d(window.clone(), conv.weight.val(), bias, options));
            buffers.push(window.narrow(2, 1, keep));
        }
        let out = self.gru.forward(h.swap_dims(1, 2), Some(state.h));
        let h_new = out.reshape([1, H]);
        let logit = self.head.forward(h_new.clone());
        (
            logit,
            State {
                convs: buffers,
                h: h_new,
            },
        )
    }
}

// --- the same network as plain slices: what a firmware loop would hold ---
struct PlainConv {
    w: Vec<f32>, // [c_out, c_in, k]
    b: Vec<f32>,
    c_in: usize,
    d: usize,
    ring: Vec<f32>, // [d (k - 1)][c_in]
    pos: usize,
}

struct Plain {
    convs: Vec<PlainConv>,
    w_x: [Vec<f32>; 3], // per gate (r, z, n): [C, H], Burn's [in, out] layout
    w_h: [Vec<f32>; 3], // per gate: [H, H]
    head: Vec<f32>,     // [H]
    h: Vec<f32>,        // the GRU state
}

impl Plain {
    fn from(model: &Detector) -> Self {
        let convs = model
            .convs
            .iter()
            .map(|c| {
                let c_in = c.weight.dims()[1];
                let len = c.dilation * (K - 1);
                PlainConv {
                    w: to_vec(c.weight.val()),
                    b: to_vec(c.bias.as_ref().unwrap().val()),
                    c_in,
                    d: c.dilation,
                    ring: vec![0.0; len * c_in],
                    pos: 0,
                }
            })
            .collect();
        let g = &model.gru;
        let gates = [&g.reset_gate, &g.update_gate, &g.new_gate];
        Self {
            convs,
            w_x: gates.map(|g| to_vec(g.input_transform.weight.val())),
            w_h: gates.map(|g| to_vec(g.hidden_transform.weight.val())),
            head: to_vec(model.head.weight.val()),
            h: vec![0.0; H],
        }
        // (every bias in this model is zero, so the plain version leaves them out)
    }

    /// Back to the zero state: a new clip starts with silence behind it.
    fn reset(&mut self) {
        for c in &mut self.convs {
            c.ring.iter_mut().for_each(|v| *v = 0.0);
            c.pos = 0;
        }
        self.h.iter_mut().for_each(|v| *v = 0.0);
    }

    fn step(&mut self, x_t: f32) -> f32 {
        let mut v = vec![x_t];
        for c in &mut self.convs {
            let len = c.ring.len() / c.c_in;
            let c_out = c.b.len();
            let mut y = c.b.clone();
            for (o, y_o) in y.iter_mut().enumerate() {
                for i in 0..c.c_in {
                    for u in 0..K {
                        let back = c.d * (K - 1 - u);
                        let x = if back == 0 {
                            v[i]
                        } else {
                            c.ring[((c.pos + len - back) % len) * c.c_in + i]
                        };
                        *y_o += c.w[(o * c.c_in + i) * K + u] * x;
                    }
                }
                *y_o = y_o.max(0.0); // ReLU
            }
            c.ring[c.pos * c.c_in..(c.pos + 1) * c.c_in].copy_from_slice(&v);
            c.pos = (c.pos + 1) % len;
            debug_assert_eq!(y.len(), c_out);
            v = y;
        }
        // GRU: a = x W_x, b = h W_h for each gate ([in, out] layout).
        let mut a = [[0.0f32; H]; 3];
        let mut b = [[0.0f32; H]; 3];
        for g in 0..3 {
            for j in 0..H {
                a[g][j] = (0..C).map(|i| v[i] * self.w_x[g][i * H + j]).sum();
                b[g][j] = (0..H).map(|i| self.h[i] * self.w_h[g][i * H + j]).sum();
            }
        }
        let sigmoid = |s: f32| 1.0 / (1.0 + (-s).exp());
        for j in 0..H {
            let r = sigmoid(a[0][j] + b[0][j]);
            let z = sigmoid(a[1][j] + b[1][j]);
            let n = (a[2][j] + r * b[2][j]).tanh();
            self.h[j] = (1.0 - z) * n + z * self.h[j];
        }
        (0..H).map(|j| self.h[j] * self.head[j]).sum()
    }
}

/// The formula budget of Section 23.5.
struct Budget {
    conv_macs: usize, // sum of c_out c_in k
    gru_macs: usize,  // 3 H (C + H)
    head_macs: usize, // H
    params: usize,
    conv_state: usize, // sum of c_in d (k - 1)
    gru_state: usize,  // H
}

fn budget() -> Budget {
    let mut b = Budget {
        conv_macs: 0,
        gru_macs: 3 * H * (C + H), // three gates, input and hidden matmuls
        head_macs: H,
        params: 3 * (C * H + H + H * H + H) + H + 1, // GRU, head
        conv_state: 0,
        gru_state: H,
    };
    for (i, &d) in DILATIONS.iter().enumerate() {
        let c_in = if i == 0 { 1 } else { C };
        b.conv_macs += C * c_in * K; //        one output frame
        b.params += C * c_in * K + C; //       weight + bias
        b.conv_state += c_in * d * (K - 1); // the ring buffer
    }
    b
}

fn clip(rng: &mut Lcg) -> Vec<f32> {
    rng.grid(T, 32, 64.0) // a noise clip is enough for timing
}

#[allow(dead_code)] // the fields are read by the test
struct Results {
    num_params: usize,
    whole: Vec<f32>,
    burn_stream: Vec<f32>,
    plain_stream: Vec<f32>,
}

fn build(timing: bool) -> Results {
    let device = Device::flex();
    let model = Detector::new(&device);
    let b = budget();
    let macs = b.conv_macs + b.gru_macs + b.head_macs;
    let (params, state) = (b.params, b.conv_state + b.gru_state);
    let num_params = model.num_params();
    println!(
        "per frame: {macs} MACs (convs {}, GRU {}, head {})",
        b.conv_macs, b.gru_macs, b.head_macs
    );
    println!(
        "parameters: formula {params}, num_params() {num_params} = {} bytes f32",
        4 * params
    );
    println!(
        "state: {state} floats = {} bytes (conv rings 98, GRU 16)",
        4 * state
    );
    println!(
        "recompute a {T}-frame window instead: {} MACs per frame ({T}x)",
        T * macs
    );

    let mut rng = Lcg(99);
    let x = clip(&mut rng);
    let xt = Tensor::<3>::from_data(TensorData::new(x.clone(), [1, 1, T]), &device);

    // The four ways, on the same clip.
    let whole = to_vec(model.forward(xt.clone()));
    let mut st = model.zero_state(&device);
    let mut burn_stream = Vec::new();
    for t in 0..T {
        let (y, next) = model.step(xt.clone().narrow(2, t, 1), st);
        burn_stream.extend(to_vec(y));
        st = next;
    }
    let mut plain = Plain::from(&model);
    let plain_stream: Vec<f32> = x.iter().map(|&v| plain.step(v)).collect();
    let diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f32::max)
    };
    println!(
        "max|whole - Burn stream| = {:.1e}, max|whole - plain stream| = {:.1e}",
        diff(&whole, &burn_stream),
        diff(&whole, &plain_stream)
    );

    if timing {
        let clips: Vec<Vec<f32>> = (0..CLIPS).map(|_| clip(&mut rng)).collect();
        let frames = (CLIPS * T) as f64;
        let us = |start: Instant, n: f64| start.elapsed().as_secs_f64() * 1e6 / n;

        let _ = to_vec(model.forward(xt.clone())); // warm-up: first-call costs out of the way
        let start = Instant::now();
        for c in &clips {
            let xt = Tensor::<3>::from_data(TensorData::new(c.clone(), [1, 1, T]), &device);
            let _ = to_vec(model.forward(xt));
        }
        let t_whole = us(start, frames);

        // Recompute: at every frame, rerun the model on the last T frames (zeros before the clip).
        let start = Instant::now();
        let c = &clips[0];
        for t in 0..T {
            let mut window = vec![0.0f32; T];
            window[T - 1 - t..].copy_from_slice(&c[..=t]);
            let xt = Tensor::<3>::from_data(TensorData::new(window, [1, 1, T]), &device);
            let _ = to_vec(model.forward(xt));
        }
        let t_recompute = us(start, T as f64);

        let start = Instant::now();
        for c in &clips {
            let mut st = model.zero_state(&device);
            for &v in c {
                let xt = Tensor::<3>::from_data(TensorData::new(vec![v], [1, 1, 1]), &device);
                let (y, next) = model.step(xt, st);
                let _ = to_vec(y);
                st = next;
            }
        }
        let t_burn = us(start, frames);

        let start = Instant::now();
        let mut sink = 0.0;
        let mut plain = Plain::from(&model);
        for _ in 0..50 {
            for c in &clips {
                plain.reset();
                for &v in c {
                    sink += plain.step(v);
                }
            }
        }
        let t_plain = us(start, 50.0 * frames);
        std::hint::black_box(sink);

        println!("time per frame on this CPU (flex, batch 1):");
        println!("  whole clip, amortised  {t_whole:>9.1} us");
        println!("  recompute the window   {t_recompute:>9.1} us");
        println!("  stream, Burn           {t_burn:>9.1} us");
        println!("  stream, plain Rust     {t_plain:>9.2} us");
    }

    Results {
        num_params,
        whole,
        burn_stream,
        plain_stream,
    }
}

fn main() {
    build(true);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let r = build(false);
        let b = budget();
        assert_eq!((b.conv_macs, b.gru_macs, b.head_macs), (408, 1152, 16));
        assert_eq!((b.params, b.conv_state, b.gru_state), (1697, 98, 16));
        assert_eq!(r.num_params, 1697);
        // main.py: the same untrained model on the same clip, first five logits.
        let expected = [
            -0.000382167229,
            -0.000360569509,
            -0.000787793775,
            -0.00118244661,
            -0.00146851013,
        ];
        for (a, b) in r.whole.iter().zip(expected) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
        for v in [&r.burn_stream, &r.plain_stream] {
            let d = r
                .whole
                .iter()
                .zip(v.iter())
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f32::max);
            assert!(d < 1e-6, "{d}");
        }
    }
}
