use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d,
};
use burn::tensor::activation::silu;
use burn::tensor::{Device, Int, Tensor, TensorData};

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
    fn uniform(&mut self) -> f32 {
        (self.next() as f32 + 0.5) / 32768.0 // in (0, 1), exact in f32
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

// One training example of flow matching: data x1, noise x0, a time t, a label.
#[derive(Clone, Debug)]
struct FlowItem {
    x1: Vec<f32>,
    x0: Vec<f32>,
    t: f32,
    label: usize,
}

// Item `index` is a pure function of (seed, index): a fresh image, fresh noise and a fresh t.
fn flow_item(seed: u32, index: usize, p_drop: u32) -> FlowItem {
    let mut rng = Lcg(seed ^ (index as u32).wrapping_mul(2654435761));
    let class = index % 4;
    let x1 = to_signed(make_image(class, &mut rng));
    let x0 = rng.normals(PIXELS);
    let t = rng.uniform();
    let drop = rng.below(100) < p_drop as usize; // p_drop percent of labels -> "no label"
    FlowItem {
        x1,
        x0,
        t,
        label: if drop { NULL_CLASS } else { class },
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

// --- fixed starting weights, so every number can be checked against PyTorch ---
// Chapter 13's formula, w_i = sin(2.3 i + phase) / sqrt(fan_in), with a new phase per tensor.
// Biases start at zero; GroupNorm keeps its default gamma = 1, beta = 0.
fn det(n: usize, fan_in: usize, phase: f64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f64 * 2.3 + phase).sin() / (fan_in as f64).sqrt()) as f32)
        .collect()
}

fn param<const D: usize>(v: Vec<f32>, shape: [usize; D], device: &Device) -> Param<Tensor<D>> {
    Param::from_tensor(Tensor::from_data(TensorData::new(v, shape), device))
}

struct Fixed<'a> {
    k: usize, // tensors filled so far: the phase is 0.1 * k
    device: &'a Device,
}

impl Fixed<'_> {
    fn phase(&mut self) -> f64 {
        self.k += 1;
        0.1 * self.k as f64
    }
    fn conv(&mut self, conv: &mut Conv2d) {
        let [o, i, kh, kw] = conv.weight.val().dims();
        let w = det(o * i * kh * kw, i * kh * kw, self.phase());
        conv.weight = param(w, [o, i, kh, kw], self.device);
        conv.bias = Some(param(vec![0.0; o], [o], self.device));
    }
    fn conv_t(&mut self, conv: &mut ConvTranspose2d) {
        let [i, o, kh, kw] = conv.weight.val().dims(); // [in, out, k, k], as in PyTorch
        conv.weight = param(
            det(i * o * kh * kw, i, self.phase()),
            [i, o, kh, kw],
            self.device,
        );
        conv.bias = Some(param(vec![0.0; o], [o], self.device));
    }
    fn linear(&mut self, lin: &mut Linear) {
        let [i, o] = lin.weight.val().dims(); // Burn stores [in, out]; PyTorch [out, in]
        let w = det(o * i, i, self.phase()); // generated in PyTorch's order...
        let w_t: Vec<f32> = (0..i * o).map(|k| w[(k % o) * i + k / o]).collect(); // ...transposed
        lin.weight = param(w_t, [i, o], self.device);
        lin.bias = Some(param(vec![0.0; o], [o], self.device));
    }
    fn block(&mut self, b: &mut ResBlock) {
        self.conv(&mut b.conv1);
        self.linear(&mut b.cond);
        self.conv(&mut b.conv2);
        if let Some(s) = b.skip.as_mut() {
            self.conv(s);
        }
    }
}

impl UNet {
    fn with_fixed_weights(mut self, device: &Device) -> Self {
        let mut f = Fixed { k: 0, device };
        f.linear(&mut self.time1);
        f.linear(&mut self.time2);
        let [n, d] = self.class_emb.weight.val().dims();
        self.class_emb.weight = param(det(n * d, 1, f.phase()), [n, d], device);
        f.conv(&mut self.conv_in);
        f.block(&mut self.enc);
        f.conv(&mut self.down);
        f.block(&mut self.mid);
        f.conv_t(&mut self.up);
        f.block(&mut self.dec);
        f.conv(&mut self.conv_out);
        self
    }
}

// Equation (params): P = 145 C^2 + 4 C d + 55 C + 1 + E d + d^2 + 7 d
fn params_formula(c: usize, e: usize, d: usize) -> usize {
    145 * c * c + 4 * c * d + 55 * c + 1 + e * d + d * d + 7 * d
}

fn batch(items: &[FlowItem], device: &Device) -> (Tensor<4>, Tensor<4>, Tensor<1>, Tensor<1, Int>) {
    let n = items.len();
    let x1: Vec<f32> = items.iter().flat_map(|it| it.x1.clone()).collect();
    let x0: Vec<f32> = items.iter().flat_map(|it| it.x0.clone()).collect();
    let t: Vec<f32> = items.iter().map(|it| it.t).collect();
    let labels: Vec<i64> = items.iter().map(|it| it.label as i64).collect();
    (
        Tensor::from_data(TensorData::new(x1, [n, 1, SIZE, SIZE]), device),
        Tensor::from_data(TensorData::new(x0, [n, 1, SIZE, SIZE]), device),
        Tensor::from_data(TensorData::new(t, [n]), device),
        Tensor::from_data(TensorData::new(labels, [n]), device),
    )
}

fn vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec().unwrap()
}

struct Report {
    num_params: usize,
    temb: Vec<f32>,  // time embedding of t = 0.5, first 4 numbers
    t: Vec<f32>,     // the four items' times
    loss: f32,       // flow-matching loss on the four items
    v: Vec<f32>,     // velocity, first 4 pixels of each item
    step: Vec<f32>,  // one guided Euler step, first 4 pixels of each item
    v_mean_abs: f32, // mean |v| over all four items
}

/// The four items 0..4 of the training stream (one per class), one forward pass,
/// the flow-matching loss, and one classifier-free-guided Euler step from fixed noise.
fn build() -> Report {
    let device = Device::flex();
    let model = UNet::new(16, 32, 64, &device).with_fixed_weights(&device);

    let temb = vec(time_embedding(Tensor::from_floats([0.5], &device), 32));

    // x_t = (1 - t) x0 + t x1, and the target velocity v = x1 - x0 (Chapter 17)
    let items: Vec<FlowItem> = (0..4).map(|i| flow_item(18, i, 0)).collect();
    let (x1, x0, t, labels) = batch(&items, &device);
    let tt = t.clone().unsqueeze_dims::<4>(&[1, 2, 3]);
    let xt = x0.clone() * (tt.clone().neg() + 1.0) + x1.clone() * tt;
    let v = model.forward(xt, t.clone(), labels);
    let loss = (v.clone() - (x1 - x0)).powi_scalar(2).mean();

    // One Euler step from pure noise (t = 0) with guidance: v = v_u + w (v_c - v_u), w = 2.
    let noise = Tensor::<4>::from_data(
        TensorData::new(Lcg(99).normals(4 * PIXELS), [4, 1, SIZE, SIZE]),
        &device,
    );
    let both = Tensor::cat(vec![noise.clone(), noise.clone()], 0); // cond and uncond, one call
    let y = Tensor::<1, Int>::from_ints([0, 1, 2, 3, 4, 4, 4, 4], &device);
    let v_both = model.forward(both, Tensor::zeros([8], &device), y);
    let v_c = v_both.clone().slice(0..4);
    let v_u = v_both.slice(4..8);
    let guided = v_u.clone() + (v_c - v_u).mul_scalar(2.0);
    let step = noise + guided.mul_scalar(1.0 / 8.0); // dt = 1/8

    let first4 = |x: Vec<f32>| -> Vec<f32> {
        (0..4)
            .flat_map(|i| x[i * PIXELS..i * PIXELS + 4].to_vec())
            .collect()
    };
    let v_all = vec(v);
    Report {
        num_params: model.num_params(),
        temb: temb[..4].to_vec(),
        t: vec(t),
        loss: vec(loss)[0],
        v_mean_abs: v_all.iter().map(|x| x.abs()).sum::<f32>() / v_all.len() as f32,
        v: first4(v_all),
        step: first4(vec(step)),
    }
}

fn show(xs: &[f32]) -> String {
    let s: Vec<String> = xs.iter().map(|x| format!("{x:.4}")).collect();
    format!("[{}]", s.join(", "))
}

fn main() {
    let r = build();
    println!("U-Net, C = 16, E = 32, d = 64");
    println!("num_params()      = {}", r.num_params);
    println!("formula (params)  = {}", params_formula(16, 32, 64));
    println!("time embedding, t = 0.5, first 4 = {}", show(&r.temb));
    println!("item times t      = {}", show(&r.t));
    for i in 0..4 {
        println!("class {i}: v[0..4]  = {}", show(&r.v[4 * i..4 * i + 4]));
    }
    println!("mean |v|          = {:.4}", r.v_mean_abs);
    println!("flow-matching loss = {:.4}", r.loss);
    println!("one Euler step from noise, w = 2, dt = 1/8:");
    for i in 0..4 {
        println!("class {i}: x[0..4]  = {}", show(&r.step[4 * i..4 * i + 4]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: &[f32], want: &[f32], tol: f32) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < tol, "{g} vs {w}");
        }
    }

    #[test]
    fn matches_pytorch() {
        let r = build();
        // From python/main.py: same weights, same items, same noise.
        assert_eq!(r.num_params, EXPECTED_PARAMS);
        assert_eq!(r.num_params, params_formula(16, 32, 64));
        close(&r.temb, &EXPECTED_TEMB, 1e-4);
        close(&r.v, &EXPECTED_V, 1e-4);
        close(&r.step, &EXPECTED_STEP, 1e-4);
        assert!((r.loss - EXPECTED_LOSS).abs() < 1e-4);
    }

    const EXPECTED_PARAMS: usize = 48689;
    const EXPECTED_TEMB: [f32; 4] = [-0.467772, -0.999998, 0.859476, 0.813053];
    const EXPECTED_V: [f32; 16] = [
        -0.370179, 1.765336, -1.197356, -0.391903, -0.104603, 0.071727, -0.012145, -0.439532,
        0.302640, -0.827440, 1.443363, -2.041215, -0.266395, 1.833725, -2.256218, 2.343906,
    ];
    const EXPECTED_STEP: [f32; 16] = [
        0.262024, 0.653188, -0.108633, -0.657549, -1.243223, 0.621175, 0.955073, -1.017622,
        -0.963196, -0.301883, 0.161812, -2.588175, 0.414550, -0.154499, -0.762592, -0.824391,
    ];
    const EXPECTED_LOSS: f32 = 2.837300;
}
