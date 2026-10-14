use burn::module::Param;
use burn::nn::gru::{Gru, GruConfig};
use burn::tensor::{Device, Tensor, TensorData};

// Chapter 23 -- A GRU, one frame at a time.
//
// A recurrent layer is already a streaming layer: its only memory is the hidden
// state h, and each frame updates it once. We write the step by hand (PyTorch's
// equations, PyTorch's weight layout), then check it against Burn's nn::gru::Gru
// run over the whole sequence and run one frame per call.

const I: usize = 3; // input features per frame
const H: usize = 4; // hidden size
const T: usize = 10; // frames

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

/// PyTorch's nn.GRU layout: rows of w_ih [3H, I] and w_hh [3H, H] are the
/// r, z and n gates, in that order.
struct GruWeights {
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// One frame: h' = (1 - z) * n + z * h.
fn gru_step(x: &[f32], h: &[f32], w: &GruWeights) -> Vec<f32> {
    // a = W_ih x + b_ih and b = W_hh h + b_hh, for all 3H gate rows at once.
    let a: Vec<f32> = (0..3 * H)
        .map(|r| w.b_ih[r] + (0..I).map(|c| w.w_ih[r * I + c] * x[c]).sum::<f32>())
        .collect();
    let b: Vec<f32> = (0..3 * H)
        .map(|r| w.b_hh[r] + (0..H).map(|c| w.w_hh[r * H + c] * h[c]).sum::<f32>())
        .collect();
    (0..H)
        .map(|j| {
            let r = sigmoid(a[j] + b[j]); //                  reset gate
            let z = sigmoid(a[H + j] + b[H + j]); //          update gate
            let n = (a[2 * H + j] + r * b[2 * H + j]).tanh(); // candidate
            (1.0 - z) * n + z * h[j]
        })
        .collect()
}

/// Copy one gate's PyTorch rows into a Burn GateController: Burn's Linear
/// stores [in, out], so each [H, in] block is transposed.
fn gate_rows(
    w: &[f32],
    b: &[f32],
    gate: usize,
    cols: usize,
    device: &Device,
) -> (Param<Tensor<2>>, Param<Tensor<1>>) {
    let block = &w[gate * H * cols..(gate + 1) * H * cols]; // [H, cols]
    let t = Tensor::<2>::from_data(TensorData::new(block.to_vec(), [H, cols]), device).transpose();
    let bias = Tensor::<1>::from_data(
        TensorData::new(b[gate * H..(gate + 1) * H].to_vec(), [H]),
        device,
    );
    (Param::from_tensor(t), Param::from_tensor(bias))
}

fn burn_gru(w: &GruWeights, reset_after: bool, device: &Device) -> Gru {
    let mut gru = GruConfig::new(I, H, true)
        .with_reset_after(reset_after)
        .init(device);
    // PyTorch order r, z, n  ->  Burn's reset_gate, update_gate, new_gate.
    for (gate, ctrl) in [
        (0, &mut gru.reset_gate),
        (1, &mut gru.update_gate),
        (2, &mut gru.new_gate),
    ] {
        let (wi, bi) = gate_rows(&w.w_ih, &w.b_ih, gate, I, device);
        let (wh, bh) = gate_rows(&w.w_hh, &w.b_hh, gate, H, device);
        ctrl.input_transform.weight = wi;
        ctrl.input_transform.bias = Some(bi);
        ctrl.hidden_transform.weight = wh;
        ctrl.hidden_transform.bias = Some(bh);
    }
    gru
}

fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec().unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f32::max)
}

#[allow(dead_code)] // the fields are read by the test
struct Results {
    by_hand: Vec<f32>,      // [T, H]: h after every frame
    burn_whole: Vec<f32>,   // Gru::forward over the whole sequence
    burn_steps: Vec<f32>,   // Gru::forward one frame per call
    reset_before: Vec<f32>, // reset_after = false (not PyTorch's GRU)
}

fn build() -> Results {
    let device = Device::flex();
    let mut rng = Lcg(2005);
    let w = GruWeights {
        w_ih: rng.grid(3 * H * I, 16, 32.0),
        w_hh: rng.grid(3 * H * H, 16, 32.0),
        b_ih: rng.grid(3 * H, 16, 32.0),
        b_hh: rng.grid(3 * H, 16, 32.0),
    };
    let x = rng.grid(T * I, 8, 4.0); // [T, I]

    // 1. By hand: the hidden state is the only state.
    let mut h = vec![0.0f32; H];
    let mut by_hand = Vec::new();
    for t in 0..T {
        h = gru_step(&x[t * I..(t + 1) * I], &h, &w);
        by_hand.extend_from_slice(&h);
    }

    // 2. Burn's Gru, whole sequence: [batch, T, I] -> [batch, T, H].
    let gru = burn_gru(&w, true, &device);
    let xt = Tensor::<3>::from_data(TensorData::new(x.clone(), [1, T, I]), &device);
    let burn_whole = to_vec(gru.forward(xt.clone(), None));

    // 3. Burn's Gru, one frame per call, the state passed back in.
    let mut state: Option<Tensor<2>> = None;
    let mut burn_steps = Vec::new();
    for t in 0..T {
        let y = gru.forward(xt.clone().narrow(1, t, 1), state); // [1, 1, H]
        let h_t = y.reshape([1, H]);
        burn_steps.extend(to_vec(h_t.clone()));
        state = Some(h_t);
    }

    // 4. The other GRU: reset applied before the hidden matmul.
    let reset_before = to_vec(burn_gru(&w, false, &device).forward(xt, None));

    let show = |v: &[f32]| format!("{:.6?}", &v[(T - 1) * H..]);
    println!("x: {T} frames of {I} features, hidden size {H}");
    println!("h[{}] by hand      = {}", T - 1, show(&by_hand));
    println!("h[{}] Burn, whole  = {}", T - 1, show(&burn_whole));
    println!("h[{}] Burn, steps  = {}", T - 1, show(&burn_steps));
    println!(
        "max|hand - whole|  = {:.1e}",
        max_diff(&by_hand, &burn_whole)
    );
    println!(
        "max|whole - steps| = {:.1e}",
        max_diff(&burn_whole, &burn_steps)
    );
    println!(
        "reset_after = false: max|diff| = {:.4}  (a different GRU)",
        max_diff(&burn_whole, &reset_before)
    );
    println!(
        "state = {H} floats = {} bytes; MACs per frame = 3H(I + H) = {}",
        H * 4,
        3 * H * (I + H)
    );

    Results {
        by_hand,
        burn_whole,
        burn_steps,
        reset_before,
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
        // nn.GRU in main.py, same weights, same input: the last hidden state.
        let expected = [0.149455, 0.421631, 0.50154, -0.536617];
        for (a, b) in r.by_hand[(T - 1) * H..].iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        assert!(max_diff(&r.by_hand, &r.burn_whole) < 1e-5);
        assert!(max_diff(&r.burn_whole, &r.burn_steps) < 1e-6);
        assert!(max_diff(&r.burn_whole, &r.reset_before) > 1e-3);
    }
}
