use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d,
};
use burn::tensor::activation::silu;
use burn::tensor::{Device, Int, Tensor, TensorData};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SIZE: usize = 12;
const PIXELS: usize = SIZE * SIZE;
const NULL_CLASS: usize = 4; // the "no label" class used for classifier-free guidance

// A tiny LCG so Rust and Python draw exactly the same "random" numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    // Box-Muller: two uniforms -> two standard normals (computed in f64, stored as f32)
    fn normals(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let u1 = (self.next() as f64 + 1.0) / 32768.0;
            let u2 = self.next() as f64 / 32768.0;
            let r = (-2.0 * u1.ln()).sqrt();
            let a = 2.0 * std::f64::consts::PI * u2;
            out.push((r * a.cos()) as f32);
            out.push((r * a.sin()) as f32);
        }
        out.truncate(n);
        out
    }
}

// --- the time embedding: sin/cos of t at 16 frequencies, as in a transformer ---
fn time_embedding(t: Tensor<1>, dim: usize) -> Tensor<2> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::<1>::from_floats(freqs.as_slice(), &t.device()).unsqueeze_dim(0); // [1, half]
    let args = t.unsqueeze_dim(1).mul_scalar(1000.0) * freqs; // [N, half]
    Tensor::cat(vec![args.clone().sin(), args.cos()], 1) // [N, dim]
}

// --- the U-Net ---
// A residual block that also reads the conditioning vector c (time + class).
#[derive(Module, Debug)]
struct ResBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    cond: Linear, // c -> one shift per output channel
    norm2: GroupNorm,
    conv2: Conv2d,
    skip: Option<Conv2d>, // 1x1 conv when the channel count changes
}

impl ResBlock {
    fn new(c_in: usize, c_out: usize, d_cond: usize, groups: usize, device: &Device) -> Self {
        let conv = |a, b| {
            Conv2dConfig::new([a, b], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        Self {
            norm1: GroupNormConfig::new(groups, c_in).init(device),
            conv1: conv(c_in, c_out),
            cond: LinearConfig::new(d_cond, c_out).init(device),
            norm2: GroupNormConfig::new(groups, c_out).init(device),
            conv2: conv(c_out, c_out),
            skip: (c_in != c_out).then(|| Conv2dConfig::new([c_in, c_out], [1, 1]).init(device)),
        }
    }

    fn forward(&self, x: Tensor<4>, c: Tensor<2>) -> Tensor<4> {
        let h = self.conv1.forward(silu(self.norm1.forward(x.clone())));
        let shift = self.cond.forward(silu(c)).unsqueeze_dims::<4>(&[2, 3]); // [N, C, 1, 1]
        let h = self.conv2.forward(silu(self.norm2.forward(h + shift)));
        let skip = match &self.skip {
            Some(conv) => conv.forward(x),
            None => x,
        };
        h + skip
    }
}

#[derive(Module, Debug)]
struct UNet {
    time1: Linear,
    time2: Linear,
    class_emb: Embedding, // 4 classes + the null class
    conv_in: Conv2d,
    enc: ResBlock,       //  C channels at 12x12 -> the skip connection
    down: Conv2d,        //  C -> 2C, stride 2: 12x12 -> 6x6
    mid: ResBlock,       // 2C channels at 6x6
    up: ConvTranspose2d, // 2C -> C, stride 2: 6x6 -> 12x12
    dec: ResBlock,       // [up, skip] = 2C -> C
    norm_out: GroupNorm,
    conv_out: Conv2d, // C -> 1: the velocity
    t_dim: usize,
}

impl UNet {
    fn new(c: usize, t_dim: usize, d: usize, device: &Device) -> Self {
        let g = 4; // GroupNorm groups
        let same = || PaddingConfig2d::Explicit(1, 1, 1, 1);
        Self {
            time1: LinearConfig::new(t_dim, d).init(device),
            time2: LinearConfig::new(d, d).init(device),
            class_emb: EmbeddingConfig::new(5, d).init(device),
            conv_in: Conv2dConfig::new([1, c], [3, 3])
                .with_padding(same())
                .init(device),
            enc: ResBlock::new(c, c, d, g, device),
            down: Conv2dConfig::new([c, 2 * c], [3, 3])
                .with_stride([2, 2])
                .with_padding(same())
                .init(device),
            mid: ResBlock::new(2 * c, 2 * c, d, g, device),
            up: ConvTranspose2dConfig::new([2 * c, c], [2, 2])
                .with_stride([2, 2])
                .init(device),
            dec: ResBlock::new(2 * c, c, d, g, device),
            norm_out: GroupNormConfig::new(g, c).init(device),
            conv_out: Conv2dConfig::new([c, 1], [3, 3])
                .with_padding(same())
                .init(device),
            t_dim,
        }
    }

    /// The velocity field v(x, t, y). x: [N, 1, 12, 12], t: [N], labels: [N] (4 = no label).
    fn forward(&self, x: Tensor<4>, t: Tensor<1>, labels: Tensor<1, Int>) -> Tensor<4> {
        let [n] = labels.dims();
        let temb = self
            .time2
            .forward(silu(self.time1.forward(time_embedding(t, self.t_dim))));
        let cemb = self
            .class_emb
            .forward(labels.reshape([n, 1]))
            .reshape([n, temb.dims()[1]]);
        let c = temb + cemb; // one conditioning vector per image

        let h = self.conv_in.forward(x);
        let skip = self.enc.forward(h, c.clone()); // [N, C, 12, 12]
        let h = self.mid.forward(self.down.forward(skip.clone()), c.clone()); // [N, 2C, 6, 6]
        let h = Tensor::cat(vec![self.up.forward(h), skip], 1); // [N, 2C, 12, 12]
        let h = self.dec.forward(h, c);
        self.conv_out.forward(silu(self.norm_out.forward(h)))
    }
}

// --- sampling: Euler steps along the learned velocity, with classifier-free guidance ---
// Guided: one forward pass of 2n images per step (n conditioned, n not).
fn sample(model: &UNet, noise: Tensor<4>, labels: &[usize], w: f32, steps: usize) -> Tensor<4> {
    let device = noise.device();
    let n = labels.len();
    let y: Vec<i64> = labels
        .iter()
        .map(|&c| c as i64)
        .chain(std::iter::repeat_n(NULL_CLASS as i64, n))
        .collect();
    let y = Tensor::<1, Int>::from_data(TensorData::new(y, [2 * n]), &device);
    let dt = 1.0 / steps as f32;
    let mut x = noise;
    for k in 0..steps {
        let t = Tensor::<1>::full([2 * n], k as f32 * dt, &device);
        let v = model.forward(Tensor::cat(vec![x.clone(), x.clone()], 0), t, y.clone());
        let v_c = v.clone().slice(0..n);
        let v_u = v.slice(n..2 * n);
        x = x + (v_u.clone() + (v_c - v_u).mul_scalar(w)).mul_scalar(dt);
    }
    x
}

// Unguided (or a reflowed model at w = 1): one forward pass of n images per step.
fn sample_cond(model: &UNet, noise: Tensor<4>, labels: &[usize], steps: usize) -> Tensor<4> {
    let device = noise.device();
    let y: Vec<i64> = labels.iter().map(|&c| c as i64).collect();
    let y = Tensor::<1, Int>::from_data(TensorData::new(y, [labels.len()]), &device);
    let dt = 1.0 / steps as f32;
    let mut x = noise;
    for k in 0..steps {
        let t = Tensor::<1>::full([labels.len()], k as f32 * dt, &device);
        x = x.clone() + model.forward(x, t, y.clone()).mul_scalar(dt);
    }
    x
}

fn weights_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights")
}

fn noise(n: usize, seed: u32, device: &Device) -> Tensor<4> {
    let z = Lcg(seed).normals(n * PIXELS);
    Tensor::from_data(TensorData::new(z, [n, 1, SIZE, SIZE]), device)
}

const N: usize = 64; // images per call
const STEPS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

/// Median of 3 timed runs, in milliseconds per image. `into_data` waits for the result,
/// which matters on a GPU, where a call returns before the work is done.
fn ms_per_image(device: &Device, steps: usize, guided: bool) -> f64 {
    let model = UNet::new(16, 32, 64, device).load_file(weights_dir().join("flow_unet"));
    let labels: Vec<usize> = (0..N).map(|i| i % 4).collect();
    let run = || {
        let z = noise(N, 5, device);
        let start = Instant::now();
        let x = if guided {
            sample(&model, z, &labels, 2.0, steps)
        } else {
            sample_cond(&model, z, &labels, steps)
        };
        let _ = x.into_data();
        start.elapsed().as_secs_f64() * 1e3 / N as f64
    };
    run(); // warm-up
    let mut times: Vec<f64> = (0..3).map(|_| run()).collect();
    times.sort_by(f64::total_cmp);
    times[1]
}

/// Rows of (device, steps, ms per image without guidance, ms per image with guidance).
fn build() -> Vec<(&'static str, usize, f64, f64)> {
    #[allow(unused_mut)]
    let mut devices = vec![("Flex", Device::flex())];
    // A GPU: `cargo run --release --features gpu` (Metal on a Mac, Vulkan/DX12 elsewhere).
    #[cfg(feature = "gpu")]
    devices.push((
        "Wgpu",
        Device::wgpu(burn::tensor::DeviceKind::DefaultDevice),
    ));

    let mut rows = Vec::new();
    for (name, device) in &devices {
        for steps in STEPS {
            let plain = ms_per_image(device, steps, false);
            let guided = ms_per_image(device, steps, true);
            rows.push((*name, steps, plain, guided));
        }
    }
    rows
}

/// Least-squares line through (steps, ms): returns (fixed ms, ms per extra step).
fn fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
    let (mx, my) = (sx / n, sy / n);
    let sxy: f64 = points.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = points.iter().map(|(x, _)| (x - mx) * (x - mx)).sum();
    let slope = sxy / sxx;
    (my - slope * mx, slope)
}

fn main() {
    println!("sampling {N} images, median of 3 runs");
    println!("device  steps   ms/image (w = 1)   ms/image (guided)");
    let rows = build();
    for (name, steps, plain, guided) in &rows {
        println!("{name:<6} {steps:>6} {plain:>18.3} {guided:>19.3}");
    }
    let flex: Vec<_> = rows.iter().filter(|r| r.0 == "Flex").collect();
    let (a, b) = fit(&flex.iter().map(|r| (r.1 as f64, r.2)).collect::<Vec<_>>());
    let (ag, bg) = fit(&flex.iter().map(|r| (r.1 as f64, r.3)).collect::<Vec<_>>());
    println!("Flex fit, w = 1:  {a:.3} ms + {b:.3} ms per step");
    println!("Flex fit, guided: {ag:.3} ms + {bg:.3} ms per step");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // Timings differ from machine to machine; the samples must not. Four images,
        // w = 1, two steps, against python/main.py.
        let device = Device::flex();
        let model = UNet::new(16, 32, 64, &device).load_file(weights_dir().join("flow_unet"));
        let x = sample_cond(&model, noise(4, 99, &device), &[0, 1, 2, 3], 2);
        let x: Vec<f32> = x.into_data().try_to_vec().unwrap();
        for (i, want) in EXPECTED.iter().enumerate() {
            assert!((x[i * PIXELS] - want).abs() < 1e-3, "image {i}");
        }
        // And guidance with w = 1 is the unguided sampler, up to rounding.
        let g = sample(&model, noise(4, 99, &device), &[0, 1, 2, 3], 1.0, 2);
        let g: Vec<f32> = g.into_data().try_to_vec().unwrap();
        assert!(x.iter().zip(&g).all(|(a, b)| (a - b).abs() < 1e-5));
    }

    const EXPECTED: [f32; 4] = [-0.750835, -0.788412, -0.704630, -0.677710];
}
