use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor, TensorData};
use std::f64::consts::PI;

// Chapter 17 -- Example 17.2: the noise-prediction loss and its gradient, by hand.
//
// eps_theta is a three-layer MLP that reads x_t and an embedding of t.
// L = (1/N) sum_i ||eps_hat_i - eps_i||^2, so dL/d eps_hat = (2/N)(eps_hat - eps),
// and Chapter 9's dense-layer rule carries it back through the network.
// Checked against Burn's autodiff (here) and torch .backward() (python/main.py).

const R: f64 = 2.0; // radius of the ring of 8 Gaussians
const SIGMA: f64 = 0.15; // standard deviation of each Gaussian
const T: usize = 100; // diffusion steps
const BETA_1: f64 = 1e-3; // the DDPM schedule (1e-4 .. 0.02 for T = 1000), rescaled to T = 100
const BETA_T: f64 = 0.2;
const K: usize = 8; // time-embedding frequencies 1, 2, 4, ..., 128 -> 16 features
const IN: usize = 2 + 2 * K; // network input: x_t and the time embedding
const H: usize = 128; // hidden width
const BATCH: usize = 256;

// --- the LCG of Chapter 11, and the draws built on it (identical in python/main.py) ---
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    fn below(&mut self, n: u32) -> usize {
        (self.next() % n) as usize
    }
    // two 15-bit draws make one 30-bit uniform in (0, 1)
    fn uniform(&mut self) -> f64 {
        let hi = self.next() as f64;
        let lo = self.next() as f64;
        (hi * 32768.0 + lo + 0.5) / 1073741824.0
    }
    // Box-Muller: two uniforms -> two independent N(0, 1) numbers
    fn gauss2(&mut self) -> [f64; 2] {
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let th = 2.0 * PI * self.uniform();
        [r * th.cos(), r * th.sin()]
    }
    // one point of the data: pick one of 8 centres on the ring, add Gaussian noise
    fn data_point(&mut self) -> [f64; 2] {
        let ang = 2.0 * PI * self.below(8) as f64 / 8.0;
        let g = self.gauss2();
        [R * ang.cos() + SIGMA * g[0], R * ang.sin() + SIGMA * g[1]]
    }
}

// beta_t, alpha_t = 1 - beta_t and abar_t = alpha_1 ... alpha_t, for t = 0..=T (abar_0 = 1)
#[allow(dead_code)] // beta and alpha are for the samplers (Example 17.3)
struct Schedule {
    beta: Vec<f64>,
    alpha: Vec<f64>,
    abar: Vec<f64>,
}

fn linear_schedule() -> Schedule {
    let (mut beta, mut alpha, mut abar) = (vec![0.0; T + 1], vec![1.0; T + 1], vec![1.0; T + 1]);
    for t in 1..=T {
        beta[t] = BETA_1 + (BETA_T - BETA_1) * (t - 1) as f64 / (T - 1) as f64;
        alpha[t] = 1.0 - beta[t];
        abar[t] = abar[t - 1] * alpha[t];
    }
    Schedule { beta, alpha, abar }
}

// sin(2^k s), cos(2^k s) for k = 0..K: the network's only way of knowing the time
fn embed(s: f64, out: &mut Vec<f32>) {
    let w = |k: usize| (1u32 << k) as f64 * s;
    out.extend((0..K).map(|k| w(k).sin() as f32));
    out.extend((0..K).map(|k| w(k).cos() as f32));
}

fn tensor(v: Vec<f32>, rows: usize, device: &Device) -> Tensor<2> {
    let cols = v.len() / rows;
    Tensor::from_data(TensorData::new(v, [rows, cols]), device)
}

fn to_vec(x: Tensor<2>) -> Vec<f32> {
    x.into_data().try_to_vec().unwrap()
}

// --- the network: three dense layers, weights [in, out] as in Chapter 9 ---
#[derive(Clone)]
struct Mlp {
    w1: Tensor<2>, // [18, 128]
    b1: Tensor<2>, // [1, 128]
    w2: Tensor<2>, // [128, 128]
    b2: Tensor<2>, // [1, 128]
    w3: Tensor<2>, // [128, 2]
    b3: Tensor<2>, // [1, 2]
}

impl Mlp {
    // PyTorch's nn.Linear initialisation, U(-1/sqrt(fan_in), 1/sqrt(fan_in)), drawn from the LCG
    fn init(rng: &mut Lcg, device: &Device) -> Mlp {
        let mut draw = |n: usize, fan_in: usize, rows: usize| {
            let bound = 1.0 / (fan_in as f64).sqrt();
            let v = (0..n).map(|_| ((2.0 * rng.uniform() - 1.0) * bound) as f32).collect();
            tensor(v, rows, device)
        };
        let (w1, b1) = (draw(IN * H, IN, IN), draw(H, IN, 1));
        let (w2, b2) = (draw(H * H, H, H), draw(H, H, 1));
        let (w3, b3) = (draw(H * 2, H, H), draw(2, H, 1));
        Mlp { w1, b1, w2, b2, w3, b3 }
    }
    fn into_vec(self) -> Vec<Tensor<2>> {
        vec![self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
    }
    fn from_vec(v: Vec<Tensor<2>>) -> Mlp {
        let [w1, b1, w2, b2, w3, b3] = v.try_into().ok().unwrap();
        Mlp { w1, b1, w2, b2, w3, b3 }
    }
}

// What the forward pass keeps for the backward pass.
struct Cache {
    h0: Tensor<2>,
    z1: Tensor<2>,
    a1: Tensor<2>,
    z2: Tensor<2>,
    a2: Tensor<2>,
}

fn forward(p: &Mlp, h0: Tensor<2>) -> (Tensor<2>, Cache) {
    let z1 = h0.clone().matmul(p.w1.clone()) + p.b1.clone();
    let a1 = relu(z1.clone());
    let z2 = a1.clone().matmul(p.w2.clone()) + p.b2.clone();
    let a2 = relu(z2.clone());
    let out = a2.clone().matmul(p.w3.clone()) + p.b3.clone();
    (out, Cache { h0, z1, a1, z2, a2 })
}

// Chapter 9's dense-layer rule, three times. dout = dL/d(network output).
fn backward(p: &Mlp, c: Cache, dout: Tensor<2>) -> Mlp {
    let w3 = c.a2.transpose().matmul(dout.clone());
    let b3 = dout.clone().sum_dim(0);
    let d2 = dout.matmul(p.w3.clone().transpose());
    let d2 = d2.mask_fill(c.z2.lower_equal_elem(0.0), 0.0); // back through relu
    let w2 = c.a1.transpose().matmul(d2.clone());
    let b2 = d2.clone().sum_dim(0);
    let d1 = d2.matmul(p.w2.clone().transpose());
    let d1 = d1.mask_fill(c.z1.lower_equal_elem(0.0), 0.0);
    let w1 = c.h0.transpose().matmul(d1.clone());
    let b1 = d1.sum_dim(0);
    Mlp { w1, b1, w2, b2, w3, b3 }
}

// The first training batch of Example 17.4, with what we print about row 0.
struct Batch {
    h0: Tensor<2>,  // network input [x_t, embed(t/T)], [N, 18]
    eps: Tensor<2>, // the noise that was added, [N, 2]
    row0: ([f64; 2], usize, [f64; 2]),
}

fn ddpm_batch(rng: &mut Lcg, s: &Schedule, device: &Device) -> Batch {
    let (mut x0, mut eps, mut ca, mut cb, mut emb) = (vec![], vec![], vec![], vec![], vec![]);
    let mut row0 = None;
    for _ in 0..BATCH {
        let x = rng.data_point();
        let t = 1 + rng.below(T as u32);
        let e = rng.gauss2();
        row0.get_or_insert((x, t, e));
        x0.extend(x.map(|v| v as f32));
        eps.extend(e.map(|v| v as f32));
        ca.push(s.abar[t].sqrt() as f32);
        cb.push((1.0 - s.abar[t]).sqrt() as f32);
        embed(t as f64 / T as f64, &mut emb);
    }
    let eps = tensor(eps, BATCH, device);
    // the closed-form noising: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps
    let xt = tensor(x0, BATCH, device) * tensor(ca, BATCH, device)
        + eps.clone() * tensor(cb, BATCH, device);
    let h0 = Tensor::cat(vec![xt, tensor(emb, BATCH, device)], 1);
    Batch { h0, eps, row0: row0.unwrap() }
}

struct Run {
    row0: ([f64; 2], usize, [f64; 2]),
    xt0: Vec<f32>,
    eps_hat0: Vec<f32>,
    dout0: Vec<f32>,
    loss: f32,
    grads: Vec<Vec<f32>>, // by hand: w1, b1, w2, b2, w3, b3
    auto_loss: f32,
    auto_grads: Vec<Vec<f32>>, // the same six, from Burn's autodiff
}

fn build() -> Run {
    let s = linear_schedule();

    // ---- by hand, on the plain flex device ----
    let device = Device::flex();
    let p = Mlp::init(&mut Lcg(17), &device);
    let b = ddpm_batch(&mut Lcg(42), &s, &device);
    let (out, cache) = forward(&p, b.h0.clone());
    let diff = out.clone() - b.eps.clone();
    let loss = (diff.clone() * diff.clone()).sum().div_scalar(BATCH as f64);
    let dout = diff.mul_scalar(2.0 / BATCH as f64); // dL/d eps_hat
    let xt0 = to_vec(b.h0.slice([0..1, 0..2]));
    let eps_hat0 = to_vec(out.slice([0..1, 0..2]));
    let dout0 = to_vec(dout.clone().slice([0..1, 0..2]));
    let grads = backward(&p, cache, dout).into_vec().into_iter().map(to_vec).collect();

    // ---- the check: the same forward pass on an autodiff device, then backward() ----
    let ad = Device::flex().autodiff();
    let q = Mlp::init(&mut Lcg(17), &ad);
    let q = Mlp::from_vec(q.into_vec().into_iter().map(|w| w.require_grad()).collect());
    let b2 = ddpm_batch(&mut Lcg(42), &s, &ad);
    let diff = forward(&q, b2.h0).0 - b2.eps;
    let auto = (diff.clone() * diff).sum().div_scalar(BATCH as f64);
    let g = auto.backward();
    let auto_grads = q.into_vec().iter().map(|w| to_vec(w.grad(&g).unwrap())).collect();

    Run {
        row0: b.row0,
        xt0,
        eps_hat0,
        dout0,
        loss: loss.into_scalar(),
        grads,
        auto_loss: auto.into_scalar(),
        auto_grads,
    }
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let pairs = a.iter().zip(b).flat_map(|(x, y)| x.iter().zip(y));
    pairs.map(|(p, q)| (p - q).abs()).fold(0.0, f32::max)
}

fn fmt(v: &[f32]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    let run = build();
    let s = linear_schedule();
    let ((x, t, e), n) = (run.row0, BATCH);
    println!("batch N = {n}; row 0:");
    println!("  x_0 = ({:.6}, {:.6})   t = {t}   eps = ({:.6}, {:.6})", x[0], x[1], e[0], e[1]);
    println!(
        "  sqrt(abar_t) = {:.6}   sqrt(1 - abar_t) = {:.6}",
        s.abar[t].sqrt(),
        (1.0 - s.abar[t]).sqrt()
    );
    println!("  x_t           = {}", fmt(&run.xt0));
    println!("  eps_hat       = {}", fmt(&run.eps_hat0));
    println!("  dL/d eps_hat  = {}", fmt(&run.dout0));
    println!("loss = {:.6}", run.loss);
    println!("dL/db3         = {}", fmt(&run.grads[5]));
    println!("dL/dW3[0, :]   = {}", fmt(&run.grads[4][..2]));
    println!("dL/db1[0..4]   = {}", fmt(&run.grads[1][..4]));
    let count: usize = run.grads.iter().map(Vec::len).sum();
    println!(
        "burn autodiff: loss {:.6}, max|hand - autodiff| over all {count} gradients = {:.1e}",
        run.auto_loss,
        max_diff(&run.grads, &run.auto_grads)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let run = build();
        // by hand vs Burn's autodiff
        assert!((run.loss - run.auto_loss).abs() < 1e-6);
        assert!(max_diff(&run.grads, &run.auto_grads) < 1e-6);
        // by hand vs torch .backward() in python/main.py
        assert!((run.loss - 1.9449306).abs() < 1e-6);
        let torch_b3 = [-0.010712076, -0.14410010];
        let torch_w3 = [-0.012951394, -0.060498059];
        let torch_b1 = [-0.0026904035, 0.0014989860, -0.0023053230, 0.0064758784];
        let pairs = run.grads[5]
            .iter()
            .zip(torch_b3)
            .chain(run.grads[4].iter().zip(torch_w3))
            .chain(run.grads[1].iter().zip(torch_b1));
        for (a, b) in pairs {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }
}
