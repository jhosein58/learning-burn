use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d,
};
use burn::tensor::activation::silu;
use burn::tensor::{Device, Int, Tensor, TensorData};
use std::path::{Path, PathBuf};

const SIZE: usize = 12;
const PIXELS: usize = SIZE * SIZE;
const NULL_CLASS: usize = 4; // the "no label" class used for classifier-free guidance

// --- the data: Chapter 13's 12x12 shapes, generated in code (identical in python/main.py) ---
// A tiny LCG so Rust and Python draw exactly the same "random" numbers.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    fn below(&mut self, n: u32) -> usize {
        (self.next() % n) as usize
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

// class 0 = horizontal bar, 1 = vertical bar, 2 = diagonal, 3 = plus sign
fn make_image(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut img = vec![0.0f32; PIXELS];
    match class {
        0 => {
            let (r, c) = (rng.below(12), rng.below(7));
            (0..6).for_each(|j| img[r * SIZE + c + j] = 1.0);
        }
        1 => {
            let (c, r) = (rng.below(12), rng.below(7));
            (0..6).for_each(|i| img[(r + i) * SIZE + c] = 1.0);
        }
        2 => {
            let (r, c) = (rng.below(7), rng.below(7));
            (0..6).for_each(|i| img[(r + i) * SIZE + c + i] = 1.0);
        }
        _ => {
            let (r, c) = (2 + rng.below(8), 2 + rng.below(8));
            for d in 0..5 {
                img[r * SIZE + c + d - 2] = 1.0;
                img[(r + d - 2) * SIZE + c] = 1.0;
            }
        }
    }
    // background noise in [0, 0.25): exact in f32, so both languages agree bit for bit
    img.iter_mut()
        .for_each(|p| *p += rng.next() as f32 / 32768.0 * 0.25);
    img
}

// Pixels in [0, 1.25) -> [-1, 1): the range the noise lives in.
fn to_signed(img: Vec<f32>) -> Vec<f32> {
    img.into_iter().map(|p| 1.6 * p - 1.0).collect()
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
// x_{k+1} = x_k + dt * [ v_u + w (v_c - v_u) ],  t_k = k dt,  dt = 1/K   (from t = 0 to t = 1)
fn sample(model: &UNet, noise: Tensor<4>, labels: &[usize], w: f32, steps: usize) -> Tensor<4> {
    let device = noise.device();
    let n = labels.len();
    // One forward pass per step does both: the first n rows conditioned, the last n not.
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

// --- judging a sample: the nearest clean shape, over every class and position ---
// A clean template is 0.8 on the shape and -0.8 elsewhere: the mean of the data's two levels.
fn templates() -> Vec<(usize, Vec<f32>)> {
    let mut out = Vec::new();
    let mut add = |class: usize, cells: Vec<usize>| {
        let mut img = vec![-0.8f32; PIXELS];
        cells.into_iter().for_each(|p| img[p] = 0.8);
        out.push((class, img));
    };
    for r in 0..12 {
        for c in 0..7 {
            add(0, (0..6).map(|j| r * SIZE + c + j).collect());
        }
    }
    for c in 0..12 {
        for r in 0..7 {
            add(1, (0..6).map(|i| (r + i) * SIZE + c).collect());
        }
    }
    for r in 0..7 {
        for c in 0..7 {
            add(2, (0..6).map(|i| (r + i) * SIZE + c + i).collect());
        }
    }
    for r in 2..10 {
        for c in 2..10 {
            let plus = (0..5).flat_map(|d| [r * SIZE + c + d - 2, (r + d - 2) * SIZE + c]);
            add(3, plus.collect());
        }
    }
    out
}

/// For each image: (class of the nearest template, mean squared distance to it).
fn judge(images: &[f32], templates: &[(usize, Vec<f32>)]) -> Vec<(usize, f32)> {
    images
        .chunks(PIXELS)
        .map(|img| {
            templates
                .iter()
                .map(|(class, t)| {
                    let d = img
                        .iter()
                        .zip(t)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>();
                    (*class, d / PIXELS as f32)
                })
                .fold(
                    (0, f32::INFINITY),
                    |best, x| if x.1 < best.1 { x } else { best },
                )
        })
        .collect()
}

/// (percent of samples whose nearest shape is the requested class, mean distance).
fn score(images: &[f32], labels: &[usize], templates: &[(usize, Vec<f32>)]) -> (f32, f32) {
    let judged = judge(images, templates);
    let hits = judged
        .iter()
        .zip(labels)
        .filter(|((c, _), l)| c == *l)
        .count();
    let dist = judged.iter().map(|(_, d)| d).sum::<f32>() / judged.len() as f32;
    (100.0 * hits as f32 / labels.len() as f32, dist)
}

// A grid of 12x12 tiles, `cols` per row, 1-pixel grey gaps, as an 8-bit PNG.
fn save_grid(images: &[f32], cols: usize, path: &Path) {
    let rows = images.len() / PIXELS / cols;
    let (w, h) = (cols * (SIZE + 1) + 1, rows * (SIZE + 1) + 1);
    let mut buf = vec![128u8; w * h];
    for (k, img) in images.chunks(PIXELS).enumerate() {
        let (r0, c0) = ((k / cols) * (SIZE + 1) + 1, (k % cols) * (SIZE + 1) + 1);
        for (p, &x) in img.iter().enumerate() {
            let v = ((x + 1.0) / 2.0).clamp(0.0, 1.0); // [-1, 1] -> [0, 1]
            buf[(r0 + p / SIZE) * w + c0 + p % SIZE] = (v * 255.0).round() as u8;
        }
    }
    image::GrayImage::from_raw(w as u32, h as u32, buf)
        .unwrap()
        .save(path)
        .unwrap();
}

fn weights_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights")
}

fn noise(n: usize, seed: u32, device: &Device) -> Tensor<4> {
    let z = Lcg(seed).normals(n * PIXELS);
    Tensor::from_data(TensorData::new(z, [n, 1, SIZE, SIZE]), device)
}

fn vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec().unwrap()
}

const SWEEP: [f32; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]; // guidance scales w
const STEPS: usize = 16; // Euler steps
const PER_CLASS: usize = 64;

/// The trained model of Example 18.2, rebuilt and loaded from its burnpack file.
fn trained(device: &Device) -> UNet {
    UNet::new(16, 32, 64, device).load_file(weights_dir().join("flow_unet"))
}

/// Rows of (w, % of samples that are the requested shape, mean distance to the nearest shape).
/// Also writes one PNG per w: one row per class, the first 8 samples of each.
fn build(out: &Path) -> Vec<(f32, f32, f32)> {
    let device = Device::flex();
    let model = trained(&device);
    let labels: Vec<usize> = (0..4 * PER_CLASS).map(|i| i % 4).collect();
    let z = noise(labels.len(), 5, &device); // the same noise for every w
    let tpl = templates();

    SWEEP
        .iter()
        .map(|&w| {
            let x = vec(sample(&model, z.clone(), &labels, w, STEPS));
            let grid: Vec<f32> = (0..4)
                .flat_map(|c| (0..8).map(move |j| (4 * j + c) * PIXELS))
                .flat_map(|start| x[start..start + PIXELS].to_vec())
                .collect();
            save_grid(&grid, 8, &out.join(format!("c18e3_w{w}.png")));
            let (hits, dist) = score(&x, &labels, &tpl);
            (w, hits, dist)
        })
        .collect()
}

/// The same score for real images from the generator: what "perfect" looks like.
fn data_score() -> (f32, f32) {
    let mut rng = Lcg(7);
    let labels: Vec<usize> = (0..4 * PER_CLASS).map(|i| i % 4).collect();
    let x: Vec<f32> = labels
        .iter()
        .flat_map(|&c| to_signed(make_image(c, &mut rng)))
        .collect();
    score(&x, &labels, &templates())
}

fn main() {
    let out = std::env::temp_dir().join("learning-burn-c18e3");
    std::fs::create_dir_all(&out).unwrap();
    println!(
        "{} samples ({PER_CLASS} per class), {STEPS} Euler steps",
        4 * PER_CLASS
    );
    println!("guidance w   right shape   distance to nearest shape");
    for (w, hits, dist) in build(&out) {
        println!("{w:>10.1}   {hits:>10.1}%   {dist:>10.4}");
    }
    let (hits, dist) = data_score();
    println!("real data    {hits:>10.1}%   {dist:>10.4}");
    println!("one PNG grid per w (rows = classes) in {}", out.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // Four samples, one per class, w = 2, 8 steps, from fixed noise: python/main.py.
        let device = Device::flex();
        let x = vec(sample(
            &trained(&device),
            noise(4, 99, &device),
            &[0, 1, 2, 3],
            2.0,
            8,
        ));
        for (i, want) in EXPECTED.iter().enumerate() {
            let got = &x[i * PIXELS..i * PIXELS + 4];
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1e-3, "sample {i}: {g} vs {w}");
            }
        }
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        assert!((mean - EXPECTED_MEAN).abs() < 1e-4);
    }

    const EXPECTED: [[f32; 4]; 4] = [
        [-0.817837, -0.765963, -0.768484, -0.816734],
        [-0.882500, -0.738397, -0.694633, -0.841591],
        [-0.876325, -0.825321, -0.700350, -0.955832],
        [-0.692059, -0.727396, -0.798984, -0.742959],
    ];
    const EXPECTED_MEAN: f32 = -0.705994;
}
