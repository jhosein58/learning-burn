use burn::tensor::{Device, Tensor, TensorData};
use std::f64::consts::PI;

// Chapter 17 -- Example 17.1: the data, the noise schedule, and the forward process.
//
// The data are 2000 points from a ring of 8 Gaussians, drawn by an LCG.
// The forward process adds Gaussian noise in T = 100 small steps. Its closed form,
// x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, jumps to any step at once.
// We run the chain step by step and check it against the closed form.

const R: f64 = 2.0; // radius of the ring of 8 Gaussians
const SIGMA: f64 = 0.15; // standard deviation of each Gaussian
const T: usize = 100; // diffusion steps
const BETA_1: f64 = 1e-3; // the DDPM schedule (1e-4 .. 0.02 for T = 1000), rescaled to T = 100
const BETA_T: f64 = 0.2;
const N: usize = 2000;

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

// The cosine schedule (Nichol and Dhariwal, 2021), for comparison: abar follows a cosine,
// beta_t = 1 - abar_t / abar_{t-1} is capped at 0.999.
fn cosine_abar() -> Vec<f64> {
    let f = |t: usize| ((t as f64 / T as f64 + 0.008) / 1.008 * PI / 2.0).cos().powi(2);
    let mut abar = vec![1.0; T + 1];
    for t in 1..=T {
        let beta = (1.0 - f(t) / f(t - 1)).min(0.999);
        abar[t] = abar[t - 1] * (1.0 - beta);
    }
    abar
}

fn tensor(v: Vec<f32>, rows: usize, device: &Device) -> Tensor<2> {
    let cols = v.len() / rows;
    Tensor::from_data(TensorData::new(v, [rows, cols]), device)
}

fn to_vec(x: Tensor<2>) -> Vec<f32> {
    x.into_data().try_to_vec().unwrap()
}

fn points(rng: &mut Lcg, n: usize, draw: fn(&mut Lcg) -> [f64; 2], device: &Device) -> Tensor<2> {
    let v = (0..n).flat_map(|_| draw(rng).map(|z| z as f32)).collect();
    tensor(v, n, device)
}

// The closed form: one jump from x_0 to x_t.
fn noised(x0: Tensor<2>, eps: Tensor<2>, abar: f64) -> Tensor<2> {
    x0.mul_scalar(abar.sqrt()) + eps.mul_scalar((1.0 - abar).sqrt())
}

// Mean and standard deviation of each column.
fn mean_std(x: &Tensor<2>) -> ([f64; 2], [f64; 2]) {
    let v = to_vec(x.clone());
    let n = (v.len() / 2) as f64;
    let mean = [0, 1].map(|j| v.iter().skip(j).step_by(2).map(|&a| a as f64).sum::<f64>() / n);
    let var = [0, 1].map(|j| {
        v.iter().skip(j).step_by(2).map(|&a| (a as f64 - mean[j]).powi(2)).sum::<f64>() / n
    });
    (mean, var.map(f64::sqrt))
}

// Write rows to $CSV_DIR/<name>.csv, if CSV_DIR is set (for the figures).
fn save_csv(name: &str, header: &str, rows: &[Vec<f64>]) {
    if let Ok(dir) = std::env::var("CSV_DIR") {
        let lines: Vec<String> = rows
            .iter()
            .map(|r| r.iter().map(|v| format!("{v:.6e}")).collect::<Vec<_>>().join(","))
            .collect();
        std::fs::write(format!("{dir}/{name}.csv"), format!("{header}\n{}\n", lines.join("\n")))
            .unwrap();
    }
}

struct Run {
    x0: Vec<f32>,
    eps: Vec<f32>,
    jumps: Vec<(usize, Vec<f32>)>, // closed-form x_t of the three points
    chain: Vec<(usize, [f64; 2], [f64; 2])>, // t, mean and std of the chain after t steps
    data_mean: [f64; 2],
    data_std: [f64; 2],
}

fn build(device: &Device) -> Run {
    let s = linear_schedule();
    let x = points(&mut Lcg(42), N, Lcg::data_point, device);
    let (data_mean, data_std) = mean_std(&x);

    // Three data points and three fixed noise vectors, jumped to t = 10, 50, 100.
    let x0 = x.clone().slice([0..3, 0..2]);
    let eps = points(&mut Lcg(7), 3, Lcg::gauss2, device);
    let jumps = [10, 50, 100]
        .iter()
        .map(|&t| (t, to_vec(noised(x0.clone(), eps.clone(), s.abar[t]))))
        .collect();

    // The chain, one small step at a time: x_t = sqrt(alpha_t) x_{t-1} + sqrt(beta_t) eps_t.
    let mut rng = Lcg(8);
    let mut xt = x.clone();
    let mut chain = Vec::new();
    let mut snapshots = vec![(0, to_vec(x.clone()))];
    for t in 1..=T {
        let e = points(&mut rng, N, Lcg::gauss2, device);
        xt = xt.mul_scalar(s.alpha[t].sqrt()) + e.mul_scalar(s.beta[t].sqrt());
        if [10, 25, 50, 75, 100].contains(&t) {
            let (m, sd) = mean_std(&xt);
            chain.push((t, m, sd));
            snapshots.push((t, to_vec(xt.clone())));
        }
    }
    // For the figures: 1000 points of each snapshot, and the two schedules.
    let rows: Vec<Vec<f64>> = snapshots
        .iter()
        .flat_map(|(t, v)| {
            v.chunks(2).take(1000).map(move |p| vec![*t as f64, p[0] as f64, p[1] as f64])
        })
        .collect();
    save_csv("c17e1_noising", "t,x,y", &rows);
    let cos = cosine_abar();
    let rows: Vec<Vec<f64>> =
        (0..=T).map(|t| vec![t as f64, s.beta[t], s.abar[t], cos[t]]).collect();
    save_csv("c17e1_schedule", "t,beta,abar_linear,abar_cosine", &rows);

    Run { x0: to_vec(x0), eps: to_vec(eps), jumps, chain, data_mean, data_std }
}

fn fmt(v: &[f32]) -> String {
    let items: Vec<String> = v.chunks(2).map(|p| format!("({:.4}, {:.4})", p[0], p[1])).collect();
    items.join(" ")
}

fn main() {
    let device = Device::flex();
    let run = build(&device);
    let s = linear_schedule();
    let cos = cosine_abar();
    println!("data: {N} points, ring of 8 Gaussians, radius {R:.1}, sigma {SIGMA}");
    println!(
        "  mean ({:.4}, {:.4})  std ({:.4}, {:.4})",
        run.data_mean[0], run.data_mean[1], run.data_std[0], run.data_std[1]
    );
    println!("schedule: linear beta from {BETA_1} to {BETA_T}, T = {T}");
    println!("    t   beta_t  sqrt(abar_t)  sqrt(1-abar_t)  abar_t     abar_t (cosine)");
    for t in [1, 10, 25, 50, 75, 90, 100] {
        println!(
            "  {t:>3}   {:.4}  {:.6}      {:.6}        {:.3e}  {:.3e}",
            s.beta[t],
            s.abar[t].sqrt(),
            (1.0 - s.abar[t]).sqrt(),
            s.abar[t],
            cos[t]
        );
    }
    println!("closed form, three points, fixed noise:");
    println!("  x_0    = {}", fmt(&run.x0));
    println!("  eps    = {}", fmt(&run.eps));
    for (t, v) in &run.jumps {
        println!("  x_{t:<3}  = {}", fmt(v));
    }
    println!("the chain, step by step ({N} points), against the closed form:");
    println!("    t   mean (chain)        std (chain)       std (closed form)");
    for (t, m, sd) in &run.chain {
        let ab = s.abar[*t];
        let pred = run.data_std.map(|d| (ab * d * d + 1.0 - ab).sqrt());
        println!(
            "  {t:>3}   ({:>7.4}, {:>7.4})  ({:.4}, {:.4})  ({:.4}, {:.4})",
            m[0], m[1], sd[0], sd[1], pred[0], pred[1]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let run = build(&Device::flex());
        // x_100 of the three points, and the chain's std after 100 steps, from python/main.py
        let torch_x100 =
            [-0.49216002, 0.89037782, 0.11220753, -0.02384296, -0.92908782, 1.40687919];
        for (a, b) in run.jumps[2].1.iter().zip(torch_x100) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        let torch_std100 = [0.99976702, 0.99686945];
        for (a, b) in run.chain[4].2.iter().zip(torch_std100) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
