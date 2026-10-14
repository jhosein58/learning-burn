use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor, TensorData};
use std::f64::consts::PI;

// Chapter 17 -- Example 17.3: one DDPM step and one DDIM step, with fixed inputs.
//
// The network has its fixed LCG weights (untrained), so every number is reproducible.
// The inputs are Example 17.1's three points at t = 50. Then the same two steps with an
// "oracle" eps_hat = the true eps, which must land exactly where the algebra says.

const R: f64 = 2.0; // radius of the ring of 8 Gaussians
const SIGMA: f64 = 0.15; // standard deviation of each Gaussian
const T: usize = 100; // diffusion steps
const BETA_1: f64 = 1e-3; // the DDPM schedule (1e-4 .. 0.02 for T = 1000), rescaled to T = 100
const BETA_T: f64 = 0.2;
const K: usize = 8; // time-embedding frequencies 1, 2, 4, ..., 128 -> 16 features
const IN: usize = 2 + 2 * K; // network input: x_t and the time embedding
const H: usize = 128; // hidden width

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
}

// What the forward pass keeps for the backward pass (Example 17.2; unused here).
#[allow(dead_code)]
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

fn eps_model(p: &Mlp, x: Tensor<2>, t: usize, device: &Device) -> Tensor<2> {
    let n = x.dims()[0];
    let mut emb = Vec::with_capacity(n * 2 * K);
    for _ in 0..n {
        embed(t as f64 / T as f64, &mut emb);
    }
    forward(p, Tensor::cat(vec![x, tensor(emb, n, device)], 1)).0
}

// One DDPM ancestral step t -> t-1, with fresh noise z.
fn ddpm_step(s: &Schedule, x: Tensor<2>, eps_hat: Tensor<2>, t: usize, z: Tensor<2>) -> Tensor<2> {
    let c = s.beta[t] / (1.0 - s.abar[t]).sqrt();
    let mean = (x - eps_hat.mul_scalar(c)).div_scalar(s.alpha[t].sqrt());
    // the posterior variance beta_t (1 - abar_{t-1}) / (1 - abar_t); it is 0 at t = 1
    let var = s.beta[t] * (1.0 - s.abar[t - 1]) / (1.0 - s.abar[t]);
    mean + z.mul_scalar(var.sqrt())
}

// One DDIM step t -> prev (eta = 0): estimate x_0, then re-noise it to level prev
// with the same noise. Returns (x0_hat, x_prev).
fn ddim_step(
    s: &Schedule,
    x: Tensor<2>,
    eps_hat: Tensor<2>,
    t: usize,
    prev: usize,
) -> (Tensor<2>, Tensor<2>) {
    let x0_hat =
        (x - eps_hat.clone().mul_scalar((1.0 - s.abar[t]).sqrt())).div_scalar(s.abar[t].sqrt());
    let x_prev = x0_hat.clone().mul_scalar(s.abar[prev].sqrt())
        + eps_hat.mul_scalar((1.0 - s.abar[prev]).sqrt());
    (x0_hat, x_prev)
}

fn points(rng: &mut Lcg, n: usize, draw: fn(&mut Lcg) -> [f64; 2], device: &Device) -> Tensor<2> {
    let v = (0..n).flat_map(|_| draw(rng).map(|z| z as f32)).collect();
    tensor(v, n, device)
}

fn max_abs(a: Tensor<2>, b: Tensor<2>) -> f32 {
    (a - b).abs().max().into_scalar()
}

struct Run {
    xt: Vec<f32>,
    eps_hat: Vec<f32>,
    z: Vec<f32>,
    x49: Vec<f32>,
    x0_hat: Vec<f32>,
    x40: Vec<f32>,
    oracle: [f32; 3], // max differences of the three oracle checks
}

fn build() -> Run {
    let device = Device::flex();
    let s = linear_schedule();
    let (t, prev) = (50, 40);
    // Example 17.1's three points and fixed noise, jumped to t = 50 by the closed form
    let x0 = points(&mut Lcg(42), 3, Lcg::data_point, &device);
    let eps = points(&mut Lcg(7), 3, Lcg::gauss2, &device);
    let xt =
        x0.clone().mul_scalar(s.abar[t].sqrt()) + eps.clone().mul_scalar((1.0 - s.abar[t]).sqrt());

    let p = Mlp::init(&mut Lcg(17), &device);
    let eps_hat = eps_model(&p, xt.clone(), t, &device);
    let z = points(&mut Lcg(8), 3, Lcg::gauss2, &device);
    let x49 = ddpm_step(&s, xt.clone(), eps_hat.clone(), t, z.clone());
    let (x0_hat, x40) = ddim_step(&s, xt.clone(), eps_hat.clone(), t, prev);

    // Oracle: eps_hat = eps. DDIM must give back x_0, and land on the closed-form x_40;
    // DDPM's mean (z = 0) must equal the posterior mean of q(x_{t-1} | x_t, x_0).
    let (o_x0, o_x40) = ddim_step(&s, xt.clone(), eps.clone(), t, prev);
    let x40_closed = x0.clone().mul_scalar(s.abar[prev].sqrt())
        + eps.clone().mul_scalar((1.0 - s.abar[prev]).sqrt());
    let o_mean = ddpm_step(&s, xt.clone(), eps.clone(), t, eps.zeros_like());
    let c0 = s.abar[t - 1].sqrt() * s.beta[t] / (1.0 - s.abar[t]);
    let ct = s.alpha[t].sqrt() * (1.0 - s.abar[t - 1]) / (1.0 - s.abar[t]);
    let posterior = x0.clone().mul_scalar(c0) + xt.clone().mul_scalar(ct);
    let oracle = [max_abs(o_x0, x0), max_abs(o_x40, x40_closed), max_abs(o_mean, posterior)];

    Run {
        xt: to_vec(xt),
        eps_hat: to_vec(eps_hat),
        z: to_vec(z),
        x49: to_vec(x49),
        x0_hat: to_vec(x0_hat),
        x40: to_vec(x40),
        oracle,
    }
}

fn fmt(v: &[f32]) -> String {
    let items: Vec<String> = v.chunks(2).map(|p| format!("({:.4}, {:.4})", p[0], p[1])).collect();
    items.join(" ")
}

fn main() {
    let run = build();
    let s = linear_schedule();
    let (t, prev) = (50, 40);
    println!("x_50     = {}", fmt(&run.xt));
    println!("eps_hat  = {}   (fixed LCG weights, untrained)", fmt(&run.eps_hat));
    println!("DDPM step 50 -> 49:");
    let var = s.beta[t] * (1.0 - s.abar[t - 1]) / (1.0 - s.abar[t]);
    println!(
        "  1/sqrt(alpha_t) = {:.6}   beta_t/sqrt(1-abar_t) = {:.6}   sigma_t = {:.6}",
        1.0 / s.alpha[t].sqrt(),
        s.beta[t] / (1.0 - s.abar[t]).sqrt(),
        var.sqrt()
    );
    println!("  z        = {}", fmt(&run.z));
    println!("  x_49     = {}", fmt(&run.x49));
    println!("DDIM step 50 -> 40:");
    println!(
        "  sqrt(abar_50) = {:.6}   sqrt(abar_40) = {:.6}   sqrt(1-abar_40) = {:.6}",
        s.abar[t].sqrt(),
        s.abar[prev].sqrt(),
        (1.0 - s.abar[prev]).sqrt()
    );
    println!("  x0_hat   = {}", fmt(&run.x0_hat));
    println!("  x_40     = {}", fmt(&run.x40));
    println!("oracle (eps_hat = the true eps), max |difference|:");
    println!("  DDIM x0_hat vs x_0                    {:.1e}", run.oracle[0]);
    println!("  DDIM x_40 vs closed-form x_40         {:.1e}", run.oracle[1]);
    println!("  DDPM mean vs posterior mean mu~_50    {:.1e}", run.oracle[2]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let run = build();
        // x_49 and x_40 from torch in python/main.py
        let torch_x49 =
            [-0.42829746, 1.82075202, -0.25721037, -0.28487337, -1.08297169, 1.62019134];
        let torch_x40 =
            [-0.13824841, 1.91570556, -0.49128374, -0.70782411, -2.27737999, 2.26543832];
        let pairs = run.x49.iter().zip(torch_x49).chain(run.x40.iter().zip(torch_x40));
        for (a, b) in pairs {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        assert!(run.oracle.iter().all(|&d| d < 1e-5));
    }
}
