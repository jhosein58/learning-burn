use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor, TensorData};
use std::f64::consts::PI;

// Chapter 17 -- Example 17.5: flow matching by hand.
//
// The straight path x_t = (1 - t) x_0 + t x_1 from noise x_0 to data x_1 has velocity
// x_1 - x_0. The same MLP learns it: L = (1/N) sum_i ||v_hat_i - (x_1 - x_0)_i||^2,
// whose gradient goes through the same backward pass as Example 17.2.
// Then one Euler step and one Heun step of dx/dt = v(x, t), with fixed inputs.

const R: f64 = 2.0; // radius of the ring of 8 Gaussians
const SIGMA: f64 = 0.15; // standard deviation of each Gaussian
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

// The first training batch of Example 17.6, with what we print about row 0.
struct Batch {
    h0: Tensor<2>, // network input [x_t, embed(t)], [N, 18]
    v: Tensor<2>,  // the target velocity x_1 - x_0, [N, 2]
    row0: ([f64; 2], f64, [f64; 2]),
}

// x_1 from the data, t uniform on (0, 1), x_0 ~ N(0, I).
fn fm_batch(rng: &mut Lcg, device: &Device) -> Batch {
    let (mut x0, mut x1, mut a, mut b, mut emb) = (vec![], vec![], vec![], vec![], vec![]);
    let mut row0 = None;
    for _ in 0..BATCH {
        let data = rng.data_point();
        let t = rng.uniform();
        let z = rng.gauss2();
        row0.get_or_insert((data, t, z));
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
    let h0 = Tensor::cat(vec![xt, tensor(emb, BATCH, device)], 1);
    Batch { h0, v, row0: row0.unwrap() }
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

fn points(rng: &mut Lcg, n: usize, draw: fn(&mut Lcg) -> [f64; 2], device: &Device) -> Tensor<2> {
    let v = (0..n).flat_map(|_| draw(rng).map(|z| z as f32)).collect();
    tensor(v, n, device)
}

fn max_abs(a: Tensor<2>, b: Tensor<2>) -> f32 {
    (a - b).abs().max().into_scalar()
}

struct Run {
    row0: ([f64; 2], f64, [f64; 2]),
    first: [Vec<f32>; 4], // row 0: x_t, target, v_hat, dL/dv_hat
    loss: f32,
    grads: Vec<Vec<f32>>,
    auto_loss: f32,
    auto_grads: Vec<Vec<f32>>,
    steps: [Vec<f32>; 3], // x_0.5, Euler x_0.6, Heun x_0.6
    oracle: [f32; 2],
}

fn build() -> Run {
    // ---- the loss and its gradient by hand, on the plain flex device ----
    let device = Device::flex();
    let p = Mlp::init(&mut Lcg(17), &device);
    let b = fm_batch(&mut Lcg(42), &device);
    let (out, cache) = forward(&p, b.h0.clone());
    let diff = out.clone() - b.v.clone();
    let loss = (diff.clone() * diff.clone()).sum().div_scalar(BATCH as f64);
    let dout = diff.mul_scalar(2.0 / BATCH as f64); // dL/d v_hat
    let row = |x: Tensor<2>| to_vec(x.slice([0..1, 0..2]));
    let first = [row(b.h0.clone()), row(b.v.clone()), row(out), row(dout.clone())];
    let grads = backward(&p, cache, dout).into_vec().into_iter().map(to_vec).collect();

    // ---- the check: Burn's autodiff ----
    let ad = Device::flex().autodiff();
    let q = Mlp::init(&mut Lcg(17), &ad);
    let q = Mlp::from_vec(q.into_vec().into_iter().map(|w| w.require_grad()).collect());
    let b2 = fm_batch(&mut Lcg(42), &ad);
    let diff = forward(&q, b2.h0).0 - b2.v;
    let auto = (diff.clone() * diff).sum().div_scalar(BATCH as f64);
    let g = auto.backward();
    let auto_grads = q.into_vec().iter().map(|w| to_vec(w.grad(&g).unwrap())).collect();

    // ---- one Euler and one Heun step, t = 0.5 -> 0.6, from Example 17.1's three points ----
    let x1 = points(&mut Lcg(42), 3, Lcg::data_point, &device); // data
    let x0 = points(&mut Lcg(7), 3, Lcg::gauss2, &device); // noise
    let x = (x0.clone() + x1.clone()).mul_scalar(0.5); // on the straight path at t = 0.5
    let net = |x: Tensor<2>, t: f64| v_model(&p, x, t, &device);
    let euler = euler_step(&net, x.clone(), 0.5, 0.1);
    let heun = heun_step(&net, x.clone(), 0.5, 0.1);
    // oracle: the velocity of the straight path itself; both steps must stay on it
    let straight = |_: Tensor<2>, _: f64| x1.clone() - x0.clone();
    let on_path = x0.clone().mul_scalar(0.4) + x1.clone().mul_scalar(0.6);
    let oracle = [
        max_abs(euler_step(&straight, x.clone(), 0.5, 0.1), on_path.clone()),
        max_abs(heun_step(&straight, x.clone(), 0.5, 0.1), on_path),
    ];

    Run {
        row0: b.row0,
        first,
        loss: loss.into_scalar(),
        grads,
        auto_loss: auto.into_scalar(),
        auto_grads,
        steps: [to_vec(x), to_vec(euler), to_vec(heun)],
        oracle,
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

fn fmt_pts(v: &[f32]) -> String {
    let items: Vec<String> = v.chunks(2).map(|p| format!("({:.4}, {:.4})", p[0], p[1])).collect();
    items.join(" ")
}

fn main() {
    let run = build();
    let (x1, t, x0) = run.row0;
    println!("batch N = {BATCH}; row 0:");
    println!(
        "  x_1 (data) = ({:.6}, {:.6})   t = {t:.6}   x_0 (noise) = ({:.6}, {:.6})",
        x1[0], x1[1], x0[0], x0[1]
    );
    println!("  x_t            = {}", fmt(&run.first[0]));
    println!("  x_1 - x_0      = {}", fmt(&run.first[1]));
    println!("  v_hat          = {}", fmt(&run.first[2]));
    println!("  dL/d v_hat     = {}", fmt(&run.first[3]));
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
    println!("one step of size h = 0.1, from t = 0.5 (fixed LCG weights, untrained):");
    println!("  x_0.5          = {}", fmt_pts(&run.steps[0]));
    println!("  Euler  x_0.6   = {}   (1 network call)", fmt_pts(&run.steps[1]));
    println!("  Heun   x_0.6   = {}   (2 network calls)", fmt_pts(&run.steps[2]));
    println!(
        "oracle v = x_1 - x_0, max |x_0.6 - (0.4 x_0 + 0.6 x_1)|: Euler {:.1e}  Heun {:.1e}",
        run.oracle[0], run.oracle[1]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let run = build();
        // by hand vs Burn's autodiff
        assert!((run.loss - run.auto_loss).abs() < 1e-5);
        assert!(max_diff(&run.grads, &run.auto_grads) < 1e-6);
        // by hand vs torch .backward(), and the two steps, from python/main.py
        assert!((run.loss - 6.0920348).abs() < 1e-5);
        let torch_b3 = [0.049399279, 0.16117865];
        let torch_euler =
            [0.47371492, 1.06464696, -0.72520512, -0.77912533, -1.46242058, 0.78795898];
        let torch_heun =
            [0.47029307, 1.06239915, -0.72773290, -0.78301638, -1.46555471, 0.78503275];
        let pairs = run.grads[5]
            .iter()
            .zip(torch_b3)
            .chain(run.steps[1].iter().zip(torch_euler))
            .chain(run.steps[2].iter().zip(torch_heun));
        for (a, b) in pairs {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        assert!(run.oracle.iter().all(|&d| d < 1e-6));
    }
}
