use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData};

// Chapter 23 -- A causal convolution, then the same convolution one frame at a time.
//
// Causal means "pad the left only": output t sees inputs t - d(k-1) .. t and
// nothing later. Streaming keeps the last d(k-1) input frames in a ring buffer
// and produces output t as soon as frame t arrives. With small integers as data
// every sum is exact in f32, so the two must agree to the last bit.

const T: usize = 24; // frames in the clip
const K: usize = 3; //  kernel size
const DILATIONS: [usize; 4] = [1, 2, 4, 8];
const CH: [usize; 5] = [1, 2, 2, 2, 2]; // channels: input, then after each layer

// A tiny LCG so Rust and Python draw exactly the same numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    /// A small integer in lo..=hi, as f32.
    fn int(&mut self, lo: i32, hi: i32) -> f32 {
        (lo + (self.next() % (hi - lo + 1) as u32) as i32) as f32
    }
    fn ints(&mut self, n: usize, lo: i32, hi: i32) -> Vec<f32> {
        (0..n).map(|_| self.int(lo, hi)).collect()
    }
}

/// One layer: weight [c_out, c_in, k] and bias [c_out], flat and row-major.
struct Layer {
    w: Vec<f32>,
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    d: usize,
}

/// Causal conv over a whole clip. x is [c_in, t], the result [c_out, t].
/// Tap u of output t reads input t - d (k - 1 - u); before frame 0 it reads a zero.
fn causal_conv(x: &[f32], t_len: usize, l: &Layer) -> Vec<f32> {
    let mut y = vec![0.0f32; l.c_out * t_len];
    for o in 0..l.c_out {
        for t in 0..t_len {
            let mut acc = l.b[o];
            for c in 0..l.c_in {
                for u in 0..K {
                    let back = l.d * (K - 1 - u); // how many frames into the past
                    if back > t {
                        continue; // a left-padding zero
                    }
                    acc += l.w[(o * l.c_in + c) * K + u] * x[c * t_len + t - back];
                }
            }
            y[o * t_len + t] = acc;
        }
    }
    y
}

/// The streaming form: the last d (k - 1) input frames, kept in a ring buffer.
struct StreamState {
    ring: Vec<f32>, // [len][c_in]: frame slots, oldest overwritten first
    len: usize,     // d (k - 1)
    pos: usize,     // the slot the next frame goes into (= the oldest frame)
}

impl StreamState {
    fn new(l: &Layer) -> Self {
        let len = l.d * (K - 1);
        Self {
            ring: vec![0.0; len * l.c_in], // zeros = the causal padding
            len,
            pos: 0,
        }
    }
}

/// One frame in ([c_in]), one frame out ([c_out]).
fn causal_conv_step(x_t: &[f32], l: &Layer, s: &mut StreamState) -> Vec<f32> {
    let mut y = l.b.clone();
    for (o, y_o) in y.iter_mut().enumerate() {
        for c in 0..l.c_in {
            for u in 0..K {
                let back = l.d * (K - 1 - u);
                let v = if back == 0 {
                    x_t[c] // the frame that just arrived
                } else {
                    let slot = (s.pos + s.len - back) % s.len; // `back` frames ago
                    s.ring[slot * l.c_in + c]
                };
                *y_o += l.w[(o * l.c_in + c) * K + u] * v;
            }
        }
    }
    // The new frame replaces the oldest one.
    s.ring[s.pos * l.c_in..(s.pos + 1) * l.c_in].copy_from_slice(x_t);
    s.pos = (s.pos + 1) % s.len;
    y
}

fn relu(v: &mut [f32]) {
    v.iter_mut().for_each(|a| *a = a.max(0.0));
}

fn layers(rng: &mut Lcg) -> Vec<Layer> {
    DILATIONS
        .iter()
        .enumerate()
        .map(|(i, &d)| Layer {
            w: rng.ints(CH[i + 1] * CH[i] * K, -1, 1),
            b: rng.ints(CH[i + 1], -1, 1),
            c_in: CH[i],
            c_out: CH[i + 1],
            d,
        })
        .collect()
}

/// The whole clip, layer by layer (ReLU between layers, not after the last).
fn stack_whole(x: &[f32], ls: &[Layer]) -> Vec<f32> {
    let mut h = x.to_vec();
    for (i, l) in ls.iter().enumerate() {
        h = causal_conv(&h, T, l);
        if i + 1 < ls.len() {
            relu(&mut h);
        }
    }
    h
}

/// The same stack, one frame at a time. Returns [c_out, t] like stack_whole.
fn stack_stream(x: &[f32], ls: &[Layer]) -> Vec<f32> {
    let mut states: Vec<StreamState> = ls.iter().map(StreamState::new).collect();
    let c_last = CH[CH.len() - 1];
    let mut y = vec![0.0f32; c_last * T];
    for t in 0..T {
        let mut h: Vec<f32> = (0..CH[0]).map(|c| x[c * T + t]).collect(); // frame t
        for (i, (l, s)) in ls.iter().zip(states.iter_mut()).enumerate() {
            h = causal_conv_step(&h, l, s);
            if i + 1 < ls.len() {
                relu(&mut h);
            }
        }
        for c in 0..c_last {
            y[c * T + t] = h[c];
        }
    }
    y
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f32::max)
}

#[allow(dead_code)] // the fields are read by the test
struct Results {
    one_layer: Vec<f32>,
    one_layer_burn: Vec<f32>,
    one_layer_stream: Vec<f32>,
    stack: Vec<f32>,
    stack_stream: Vec<f32>,
    first_moved: usize,
}

fn build() -> Results {
    let device = Device::flex();
    let mut rng = Lcg(14);
    let x = rng.ints(T, -2, 2); // [1, T]: one channel
    let ls = layers(&mut rng);

    // 1. One causal conv (the d = 2 layer shape, but with 1 -> 2 channels).
    let l = Layer {
        w: vec![1.0, 0.0, -1.0, 0.0, 1.0, 1.0], // [2, 1, 3]
        b: vec![0.0, 0.0],
        c_in: 1,
        c_out: 2,
        d: 2,
    };
    let one_layer = causal_conv(&x, T, &l);

    // Burn's conv1d with padding (d (k - 1), 0): all on the left.
    let pad = l.d * (K - 1);
    let xt = Tensor::<3>::from_data(TensorData::new(x.clone(), [1, 1, T]), &device);
    let wt = Tensor::<3>::from_data(TensorData::new(l.w.clone(), [2, 1, K]), &device);
    let options = ConvOptions::new_with_padding([1], [(pad, 0)], [l.d], 1);
    let yt = conv1d(xt, wt, None, options);
    println!("burn conv1d shape = {:?}", yt.dims());
    let one_layer_burn: Vec<f32> = yt.into_data().try_to_vec().unwrap();

    // The same layer, one frame at a time.
    let mut s = StreamState::new(&l);
    let mut one_layer_stream = vec![0.0f32; 2 * T];
    for t in 0..T {
        let y_t = causal_conv_step(&[x[t]], &l, &mut s);
        one_layer_stream[t] = y_t[0];
        one_layer_stream[T + t] = y_t[1];
    }

    // 2. Four dilated causal convs: whole clip vs one frame at a time.
    let stack = stack_whole(&x, &ls);
    let stack_stream = stack_stream(&x, &ls);

    // 3. Causality: change frame 10 and see which outputs move.
    let mut x2 = x.clone();
    x2[10] += 5.0;
    let moved = stack_whole(&x2, &ls);
    let first_moved = (0..T)
        .find(|&t| (0..CH[4]).any(|c| moved[c * T + t] != stack[c * T + t]))
        .unwrap();

    let fmt = |v: &[f32]| {
        v.iter()
            .map(|a| format!("{a:>3}"))
            .collect::<Vec<_>>()
            .join("")
    };
    println!("x           ={}", fmt(&x));
    println!("loops   y[0]={}", fmt(&one_layer[..T]));
    println!("loops   y[1]={}", fmt(&one_layer[T..]));
    println!(
        "max|loops - burn| = {}   max|loops - stream| = {}",
        max_diff(&one_layer, &one_layer_burn),
        max_diff(&one_layer, &one_layer_stream)
    );
    let buffers: Vec<usize> = ls.iter().map(|l| l.d * (K - 1)).collect();
    println!("stack: dilations {DILATIONS:?}, ring buffers {buffers:?} frames");
    println!("whole   y[0]={}", fmt(&stack[..T]));
    println!("stream  y[0]={}", fmt(&stack_stream[..T]));
    println!(
        "max|whole - stream| = {} over {} outputs",
        max_diff(&stack, &stack_stream),
        stack.len()
    );
    println!(
        "x[10] += 5: outputs 0..{} unchanged, output {first_moved} moves",
        first_moved - 1
    );

    Results {
        one_layer,
        one_layer_burn,
        one_layer_stream,
        stack,
        stack_stream,
        first_moved,
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
        // F.pad + F.conv1d in main.py give the same numbers.
        let expected_one = [2.0, 0.0, -1.0, 0.0, -1.0, 1.0, 3.0, 0.0];
        let expected_stack = [0.0, 9.0, 6.0, 2.0, 6.0, 7.0, 2.0, 7.0];
        assert_eq!(&r.one_layer[..8], &expected_one);
        assert_eq!(&r.stack[..8], &expected_stack);
        // Exact, not approximate: integers sum exactly in f32.
        assert_eq!(r.one_layer, r.one_layer_burn);
        assert_eq!(r.one_layer, r.one_layer_stream);
        assert_eq!(r.stack, r.stack_stream);
        assert_eq!(r.first_moved, 10);
    }
}
