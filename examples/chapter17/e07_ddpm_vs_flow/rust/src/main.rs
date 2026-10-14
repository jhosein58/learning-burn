use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor, TensorData};
use std::f64::consts::PI;
use std::time::Instant;

// Chapter 17 -- Example 17.7: DDPM and flow matching, head to head.
//
// The same data, the same network, the same weights, Adam and number of steps.
// We time the training, measure sample quality along the way at a fixed budget
// of 20 network calls, and measure how straight the two samplers' paths are.

const R: f64 = 2.0; // radius of the ring of 8 Gaussians
const SIGMA: f64 = 0.15; // standard deviation of each Gaussian
const T: usize = 100; // diffusion steps
const BETA_1: f64 = 1e-3; // the DDPM schedule (1e-4 .. 0.02 for T = 1000), rescaled to T = 100
const BETA_T: f64 = 0.2;
const K: usize = 8; // time-embedding frequencies 1, 2, 4, ..., 128 -> 16 features
const IN: usize = 2 + 2 * K; // network input: x_t and the time embedding
const H: usize = 128; // hidden width
const BATCH: usize = 256;
const STEPS: usize = 6000;
const LR: f64 = 2e-3; // Adam, decayed linearly to 0
const SAMPLES: usize = 2000;
const PATHS: usize = 64;

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
#[allow(dead_code)] // beta and alpha are for DDPM's ancestral sampler (Example 17.4)
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

// --- Adam, written out (Chapter 7), with PyTorch's defaults ---
struct Adam {
    m: Vec<Tensor<2>>,
    v: Vec<Tensor<2>>,
    t: i32,
}

impl Adam {
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;
    const EPS: f64 = 1e-8;

    fn new(p: &Mlp) -> Adam {
        let zeros: Vec<Tensor<2>> =
            p.clone().into_vec().into_iter().map(|w| w.zeros_like()).collect();
        Adam { m: zeros.clone(), v: zeros, t: 0 }
    }

    fn step(&mut self, p: Mlp, g: Mlp, lr: f64) -> Mlp {
        self.t += 1;
        let (c1, c2) = (1.0 - Self::B1.powi(self.t), 1.0 - Self::B2.powi(self.t));
        let mut out = Vec::new();
        for (i, (w, g)) in p.into_vec().into_iter().zip(g.into_vec()).enumerate() {
            let m = self.m[i].clone().mul_scalar(Self::B1) + g.clone().mul_scalar(1.0 - Self::B1);
            let v =
                self.v[i].clone().mul_scalar(Self::B2) + (g.clone() * g).mul_scalar(1.0 - Self::B2);
            let m_hat = m.clone().div_scalar(c1);
            let v_hat = v.clone().div_scalar(c2);
            out.push(w - (m_hat / v_hat.sqrt().add_scalar(Self::EPS)).mul_scalar(lr));
            (self.m[i], self.v[i]) = (m, v);
        }
        Mlp::from_vec(out)
    }
}

// One training batch: x_0 from the data, t uniform on 1..=T, eps ~ N(0, I).
// Returns the network input [x_t, embed(t/T)] and the target eps.
fn ddpm_batch(rng: &mut Lcg, s: &Schedule, device: &Device) -> (Tensor<2>, Tensor<2>) {
    let (mut x0, mut eps, mut ca, mut cb, mut emb) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..BATCH {
        let x = rng.data_point();
        let t = 1 + rng.below(T as u32);
        let e = rng.gauss2();
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
    (Tensor::cat(vec![xt, tensor(emb, BATCH, device)], 1), eps)
}

// x_1 from the data, t uniform on (0, 1), x_0 ~ N(0, I).
fn fm_batch(rng: &mut Lcg, device: &Device) -> (Tensor<2>, Tensor<2>) {
    let (mut x0, mut x1, mut a, mut b, mut emb) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..BATCH {
        let data = rng.data_point();
        let t = rng.uniform();
        let z = rng.gauss2();
        x1.extend(data.map(|v| v as f32));
        x0.extend(z.map(|v| v as f32));
        a.push((1.0 - t) as f32);
        b.push(t as f32);
        embed(t, &mut emb);
    }
    let (x0, x1) = (tensor(x0, BATCH, device), tensor(x1, BATCH, device));
    // the straight path and its velocity
    let xt = x0.clone() * tensor(a, BATCH, device) + x1.clone() * tensor(b, BATCH, device);
    let v = x1 - x0;
    (Tensor::cat(vec![xt, tensor(emb, BATCH, device)], 1), v)
}

#[derive(Clone, Copy)]
enum Kind {
    Ddpm, // target: the noise eps
    Flow, // target: the velocity x_1 - x_0
}

// Train one model, keeping a copy of the weights at each checkpoint.
// Returns the copies, the loss of every step and the seconds taken.
fn train(
    kind: Kind,
    steps: usize,
    checkpoints: &[usize],
    device: &Device,
) -> (Vec<Mlp>, Vec<f32>, f64) {
    let s = linear_schedule();
    let mut p = Mlp::init(&mut Lcg(17), device);
    let mut adam = Adam::new(&p);
    let mut rng = Lcg(42);
    let (mut saved, mut losses) = (Vec::new(), Vec::new());
    let start = Instant::now();
    for k in 0..steps {
        let (h0, target) = match kind {
            Kind::Ddpm => ddpm_batch(&mut rng, &s, device),
            Kind::Flow => fm_batch(&mut rng, device),
        };
        let (out, cache) = forward(&p, h0);
        let diff = out - target;
        let loss = (diff.clone() * diff.clone()).sum().div_scalar(BATCH as f64);
        losses.push(loss.into_scalar::<f32>());
        let g = backward(&p, cache, diff.mul_scalar(2.0 / BATCH as f64));
        p = adam.step(p, g, LR * (1.0 - k as f64 / STEPS as f64));
        if checkpoints.contains(&(k + 1)) {
            saved.push(p.clone());
        }
    }
    (saved, losses, start.elapsed().as_secs_f64())
}

// --- sampling ---
fn eps_model(p: &Mlp, x: Tensor<2>, t: usize, device: &Device) -> Tensor<2> {
    let n = x.dims()[0];
    let mut emb = Vec::with_capacity(n * 2 * K);
    for _ in 0..n {
        embed(t as f64 / T as f64, &mut emb);
    }
    forward(p, Tensor::cat(vec![x, tensor(emb, n, device)], 1)).0
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

// --- the ODE dx/dt = v(x, t), one step of size h ---
type Velocity<'a> = &'a dyn Fn(Tensor<2>, f64) -> Tensor<2>;

fn euler_step(v: Velocity, x: Tensor<2>, t: f64, h: f64) -> Tensor<2> {
    x.clone() + v(x, t).mul_scalar(h)
}

// Heun: an Euler step to look ahead, then the average of the two slopes.
fn heun_step(v: Velocity, x: Tensor<2>, t: f64, h: f64) -> Tensor<2> {
    let k1 = v(x.clone(), t);
    let k2 = v(x.clone() + k1.clone().mul_scalar(h), t + h);
    x + (k1 + k2).mul_scalar(h / 2.0)
}

fn v_model(p: &Mlp, x: Tensor<2>, t: f64, device: &Device) -> Tensor<2> {
    let n = x.dims()[0];
    let mut emb = Vec::with_capacity(n * 2 * K);
    for _ in 0..n {
        embed(t, &mut emb);
    }
    forward(p, Tensor::cat(vec![x, tensor(emb, n, device)], 1)).0
}

// --- quality: two numbers that compare a point cloud with data ---
fn data_set(seed: u32, n: usize) -> Vec<[f64; 2]> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| rng.data_point()).collect()
}

// Total-variation distance between 24x24 histograms on [-3, 3]^2 (plus one bin for outside).
fn hist_tv(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let hist = |pts: &[[f64; 2]]| {
        let mut h = vec![0.0; 24 * 24 + 1];
        for p in pts {
            let (i, j) = (((p[0] + 3.0) / 0.25).floor(), ((p[1] + 3.0) / 0.25).floor());
            let inside = (0.0..24.0).contains(&i) && (0.0..24.0).contains(&j);
            h[if inside { i as usize * 24 + j as usize } else { 24 * 24 }] +=
                1.0 / pts.len() as f64;
        }
        h
    };
    let (ha, hb) = (hist(a), hist(b));
    0.5 * ha.iter().zip(&hb).map(|(x, y)| (x - y).abs()).sum::<f64>()
}

fn noise(rng: &mut Lcg, n: usize, device: &Device) -> Tensor<2> {
    let v = (0..n).flat_map(|_| rng.gauss2().map(|z| z as f32)).collect();
    tensor(v, n, device)
}

// DDIM with `steps` network calls, from the first n points of the shared starting noise.
// Returns every intermediate x, so that we can look at the paths.
fn ddim_path(p: &Mlp, s: &Schedule, steps: usize, n: usize, device: &Device) -> Vec<Tensor<2>> {
    let mut xs = vec![noise(&mut Lcg(7), n, device)];
    for i in (1..=steps).rev() {
        let (t, prev) = (i * T / steps, (i - 1) * T / steps);
        let x = xs.last().unwrap().clone();
        let eps_hat = eps_model(p, x.clone(), t, device);
        xs.push(ddim_step(s, x, eps_hat, t, prev).1);
    }
    xs
}

// Euler or Heun with `steps` steps from t = 0 to 1, from the same starting noise.
fn flow_path(p: &Mlp, steps: usize, heun: bool, n: usize, device: &Device) -> Vec<Tensor<2>> {
    let v = |x: Tensor<2>, t: f64| v_model(p, x, t, device);
    let h = 1.0 / steps as f64;
    let mut xs = vec![noise(&mut Lcg(7), n, device)];
    for i in 0..steps {
        let (x, t) = (xs.last().unwrap().clone(), i as f64 * h);
        xs.push(if heun { heun_step(&v, x, t, h) } else { euler_step(&v, x, t, h) });
    }
    xs
}

// Total path length divided by total start-to-end distance: 1 for straight lines.
fn straightness(xs: &[Tensor<2>]) -> f64 {
    let pts: Vec<Vec<[f64; 2]>> = xs.iter().map(|x| points(x.clone())).collect();
    let (first, last) = (&pts[0], &pts[pts.len() - 1]);
    let length: f64 = pts.windows(2).map(|w| dist(&w[0], &w[1])).sum();
    length / dist(first, last)
}

// The share of the total path length covered in the first and the last 10 steps.
fn length_share(xs: &[Tensor<2>]) -> (f64, f64) {
    let pts: Vec<Vec<[f64; 2]>> = xs.iter().map(|x| points(x.clone())).collect();
    let seg: Vec<f64> = pts.windows(2).map(|w| dist(&w[0], &w[1])).collect();
    let total: f64 = seg.iter().sum();
    let first: f64 = seg[..10].iter().sum();
    let last: f64 = seg[seg.len() - 10..].iter().sum();
    (first / total, last / total)
}

// Sum over points of the distance between two clouds of the same points.
fn dist(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let d = |p: &[f64; 2], q: &[f64; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
    a.iter().zip(b).map(|(p, q)| d(p, q)).sum()
}

// Root-mean-square distance between two clouds of the same points.
fn rms(a: &Tensor<2>, b: &Tensor<2>) -> f64 {
    let d = to_vec(a.clone() - b.clone());
    (d.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / (d.len() / 2) as f64).sqrt()
}

fn points(x: Tensor<2>) -> Vec<[f64; 2]> {
    to_vec(x).chunks(2).map(|c| [c[0] as f64, c[1] as f64]).collect()
}

// Write every path as rows (path, step, x, y) to $CSV_DIR/<name>.csv, if CSV_DIR is set.
fn save_paths(name: &str, xs: &[Tensor<2>]) {
    if let Ok(dir) = std::env::var("CSV_DIR") {
        let mut rows = vec!["path,step,x,y".to_string()];
        for (k, x) in xs.iter().enumerate() {
            for (i, p) in points(x.clone()).iter().enumerate() {
                rows.push(format!("{i},{k},{:.5},{:.5}", p[0], p[1]));
            }
        }
        std::fs::write(format!("{dir}/{name}.csv"), rows.join("\n") + "\n").unwrap();
    }
}

// The first n (x_0, x_1) pairs of the flow-matching training stream: the straight lines it learns from.
fn training_pairs(n: usize, device: &Device) -> Vec<Tensor<2>> {
    let mut rng = Lcg(42);
    let (mut x0, mut x1) = (vec![], vec![]);
    for _ in 0..n {
        x1.extend(rng.data_point().map(|v| v as f32));
        rng.uniform(); // t, not needed here
        x0.extend(rng.gauss2().map(|v| v as f32));
    }
    vec![tensor(x0, n, device), tensor(x1, n, device)]
}

fn mean_last(losses: &[f32]) -> f32 {
    let tail = &losses[losses.len() - 1000..];
    tail.iter().sum::<f32>() / tail.len() as f32
}

fn main() {
    let device = Device::flex();
    let s = linear_schedule();
    let checkpoints = [250, 500, 1000, 2000, 4000, 6000];
    let (ddpm, ddpm_losses, ddpm_secs) = train(Kind::Ddpm, STEPS, &checkpoints, &device);
    let (flow, flow_losses, flow_secs) = train(Kind::Flow, STEPS, &checkpoints, &device);
    let ms = |secs: f64| 1000.0 * secs / STEPS as f64;
    println!("training, {STEPS} steps each, batch {BATCH}, the same network and Adam:");
    println!(
        "  DDPM            {:.1} ms/step   mean loss of the last 1000 steps {:.4}",
        ms(ddpm_secs),
        mean_last(&ddpm_losses)
    );
    println!(
        "  flow matching   {:.1} ms/step   mean loss of the last 1000 steps {:.4}",
        ms(flow_secs),
        mean_last(&flow_losses)
    );

    let reference = data_set(99, SAMPLES);
    let tv = |xs: Vec<Tensor<2>>| hist_tv(&points(xs.last().unwrap().clone()), &reference);
    println!("tv along the way, 20 network calls per sample (data vs data: 0.093):");
    println!("   step   DDIM 20 steps   Euler 20 steps   Heun 10 steps");
    for (k, &step) in checkpoints.iter().enumerate() {
        let a = tv(ddim_path(&ddpm[k], &s, 20, SAMPLES, &device));
        let b = tv(flow_path(&flow[k], 20, false, SAMPLES, &device));
        let c = tv(flow_path(&flow[k], 10, true, SAMPLES, &device));
        println!("  {step:>5}   {a:.3}           {b:.3}            {c:.3}");
    }

    // How far a coarse solve lands from a fine one: the discretisation error of each ODE.
    let (p, q) = (ddpm.last().unwrap(), flow.last().unwrap());
    let fine_ddim = ddim_path(p, &s, 100, SAMPLES, &device).pop().unwrap();
    let fine_flow = flow_path(q, 100, true, SAMPLES, &device).pop().unwrap();
    println!(
        "RMS distance of the end point from a 100-step solve (DDIM 100; Heun 100 for the flow):"
    );
    println!("  steps    DDIM     Euler    Heun");
    for steps in [5, 10, 20, 50] {
        let a = rms(&ddim_path(p, &s, steps, SAMPLES, &device).pop().unwrap(), &fine_ddim);
        let b = rms(&flow_path(q, steps, false, SAMPLES, &device).pop().unwrap(), &fine_flow);
        let c = rms(&flow_path(q, steps, true, SAMPLES, &device).pop().unwrap(), &fine_flow);
        println!("  {steps:>5}    {a:.4}   {b:.4}   {c:.4}");
    }

    // The paths of the first PATHS starting points, 100 steps each, and PATHS training pairs.
    let ddim = ddim_path(p, &s, 100, PATHS, &device);
    let euler = flow_path(q, 100, false, PATHS, &device);
    save_paths("c17e7_paths_ddim", &ddim);
    save_paths("c17e7_paths_flow", &euler);
    let pairs = training_pairs(PATHS, &device);
    save_paths("c17e7_pairs", &pairs);
    println!("total path length / total start-to-end distance, {PATHS} paths, 100 steps:");
    println!("  DDIM (deterministic DDPM)             {:.3}", straightness(&ddim));
    println!("  flow matching (Euler)                 {:.3}", straightness(&euler));
    println!("  flow-matching training pairs x_0 -> x_1  {:.3}", straightness(&pairs));
    println!("share of the path length covered in the first and the last 10 of the 100 steps:");
    for (name, xs) in [("DDIM         ", &ddim), ("flow matching", &euler)] {
        let (first, last) = length_share(xs);
        println!("  {name}   first {first:.3}   last {last:.3}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // The first three losses of each kind, against torch in python/main.py.
        let device = Device::flex();
        let (_, ddpm, _) = train(Kind::Ddpm, 3, &[], &device);
        let (_, flow, _) = train(Kind::Flow, 3, &[], &device);
        let torch_ddpm = [1.9449306, 1.7665430, 1.8261592];
        let torch_flow = [6.0920348, 5.7223992, 5.8955946];
        let pairs = ddpm.iter().zip(torch_ddpm).chain(flow.iter().zip(torch_flow));
        for (a, b) in pairs {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
