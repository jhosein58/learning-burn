use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor, TensorData};
use std::f64::consts::PI;
use std::time::Instant;

// Chapter 17 -- Example 17.6: train flow matching by hand, then integrate the ODE.
//
// The same MLP, the same hand-written backward pass and Adam as Example 17.4,
// the same LCG weights and the same budget: only the target changes, from the
// noise eps to the velocity x_1 - x_0. Then dx/dt = v(x, t) from t = 0 (noise)
// to t = 1 (data), with Euler (1 network call per step) and Heun (2 per step).

const R: f64 = 2.0; // radius of the ring of 8 Gaussians
const SIGMA: f64 = 0.15; // standard deviation of each Gaussian
const K: usize = 8; // time-embedding frequencies 1, 2, 4, ..., 128 -> 16 features
const IN: usize = 2 + 2 * K; // network input: x_t and the time embedding
const H: usize = 128; // hidden width
const BATCH: usize = 256;
const STEPS: usize = 6000;
const LR: f64 = 2e-3; // Adam, decayed linearly to 0
const SAMPLES: usize = 2000;

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

// Train for `steps` steps; returns the weights and the loss of every step.
fn train(steps: usize, device: &Device) -> (Mlp, Vec<f32>) {
    let mut p = Mlp::init(&mut Lcg(17), device);
    let mut adam = Adam::new(&p);
    let mut rng = Lcg(42);
    let mut losses = Vec::new();
    for k in 0..steps {
        let (h0, v) = fm_batch(&mut rng, device);
        let (out, cache) = forward(&p, h0);
        let diff = out - v;
        // L = (1/N) sum_i ||v_hat_i - (x_1 - x_0)_i||^2, and dL/d v_hat = (2/N)(v_hat - v)
        let loss = (diff.clone() * diff.clone()).sum().div_scalar(BATCH as f64);
        losses.push(loss.into_scalar::<f32>());
        let g = backward(&p, cache, diff.mul_scalar(2.0 / BATCH as f64));
        p = adam.step(p, g, LR * (1.0 - k as f64 / STEPS as f64));
    }
    (p, losses)
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

// --- sampling: integrate from t = 0 to t = 1 in `steps` equal steps ---
fn v_model(p: &Mlp, x: Tensor<2>, t: f64, device: &Device) -> Tensor<2> {
    let n = x.dims()[0];
    let mut emb = Vec::with_capacity(n * 2 * K);
    for _ in 0..n {
        embed(t, &mut emb);
    }
    forward(p, Tensor::cat(vec![x, tensor(emb, n, device)], 1)).0
}

fn noise(rng: &mut Lcg, n: usize, device: &Device) -> Tensor<2> {
    let v = (0..n).flat_map(|_| rng.gauss2().map(|z| z as f32)).collect();
    tensor(v, n, device)
}

fn flow_sample(p: &Mlp, steps: usize, heun: bool, device: &Device) -> Tensor<2> {
    let v = |x: Tensor<2>, t: f64| v_model(p, x, t, device);
    let h = 1.0 / steps as f64;
    let mut x = noise(&mut Lcg(7), SAMPLES, device); // the same starting noise as Example 17.4
    for i in 0..steps {
        let t = i as f64 * h;
        x = if heun { heun_step(&v, x, t, h) } else { euler_step(&v, x, t, h) };
    }
    x
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

// Mean distance from each point of `a` to its nearest neighbour in `b`.
fn nn_dist(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let d = |p: &[f64; 2], q: &[f64; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
    a.iter().map(|p| b.iter().map(|q| d(p, q)).fold(f64::INFINITY, f64::min)).sum::<f64>()
        / a.len() as f64
}

fn points(x: Tensor<2>) -> Vec<[f64; 2]> {
    to_vec(x).chunks(2).map(|c| [c[0] as f64, c[1] as f64]).collect()
}

// Write a point cloud to $CSV_DIR/<name>.csv, if CSV_DIR is set (for the figures).
fn save_csv(name: &str, pts: &[[f64; 2]]) {
    if let Ok(dir) = std::env::var("CSV_DIR") {
        let rows: Vec<String> = pts.iter().map(|p| format!("{:.5},{:.5}", p[0], p[1])).collect();
        let body = format!("x,y\n{}\n", rows.join("\n"));
        std::fs::write(format!("{dir}/{name}.csv"), body).unwrap();
    }
}

fn main() {
    let device = Device::flex();
    let start = Instant::now();
    let (p, losses) = train(STEPS, &device);
    let secs = start.elapsed().as_secs_f64();
    for (k, l) in losses.iter().take(5).enumerate() {
        println!("step {}  loss {l:.6}", k + 1);
    }
    for c in losses.chunks(1000).enumerate() {
        let mean = c.1.iter().sum::<f32>() / c.1.len() as f32;
        println!("steps {:>4}-{:<4}  mean loss {mean:.4}", c.0 * 1000 + 1, c.0 * 1000 + 1000);
    }
    println!("trained {STEPS} steps in {secs:.1} s ({:.1} ms/step)", 1000.0 * secs / STEPS as f64);

    let reference = data_set(99, SAMPLES);
    let other = data_set(100, SAMPLES);
    println!(
        "floor (data vs data)   tv {:.3}  nn {:.3}",
        hist_tv(&other, &reference),
        nn_dist(&other, &reference)
    );
    println!("sampler        NFE   tv     nn");
    for (name, heun) in [("Euler", false), ("Heun ", true)] {
        for steps in [1, 2, 5, 10, 20, 50, 100] {
            let x = points(flow_sample(&p, steps, heun, &device));
            let nfe = if heun { 2 * steps } else { steps };
            let (tv, nn) = (hist_tv(&x, &reference), nn_dist(&x, &reference));
            println!("{name} {steps:>3} steps {nfe:>4}  {tv:.3}  {nn:.3}");
            save_csv(&format!("c17e6_{}{steps}", name.trim().to_lowercase()), &x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // The first five training losses, from the same LCG weights and the same
        // LCG batches, against torch autograd + torch.optim.Adam in python/main.py.
        let (_, losses) = train(5, &Device::flex());
        let torch = [6.0920348, 5.7223992, 5.8955946, 5.4415960, 6.1213622];
        for (a, b) in losses.iter().zip(torch) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
