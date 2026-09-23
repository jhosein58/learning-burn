use burn::backend::NdArray;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

type MyBackend = NdArray;
const SIZE: usize = 12;

// --- the data: 12x12 "shapes", generated in code (identical in python.py) ---
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
}

// class 0 = horizontal bar, 1 = vertical bar, 2 = diagonal, 3 = plus sign
fn make_image(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut img = vec![0.0f32; SIZE * SIZE];
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

// --- the model ---
#[derive(Module, Debug)]
struct SmallCnn<B: Backend> {
    conv1: Conv2d<B>,       // 1 -> 4 channels, 3x3
    pool: MaxPool2d,        // 12x12 -> 6x6
    conv2: Conv2d<B>,       // 4 -> 8 channels, 3x3
    gap: AdaptiveAvgPool2d, // 6x6 -> 1x1: one number per channel
    head: Linear<B>,        // 8 -> 4 classes
}

impl<B: Backend> SmallCnn<B> {
    fn new(device: &B::Device) -> Self {
        let same = PaddingConfig2d::Explicit(1, 1, 1, 1); // keep 12x12 as 12x12
        Self {
            conv1: Conv2dConfig::new([1, 4], [3, 3])
                .with_padding(same.clone())
                .init(device),
            pool: MaxPool2dConfig::new([2, 2]).init(),
            conv2: Conv2dConfig::new([4, 8], [3, 3])
                .with_padding(same)
                .init(device),
            gap: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            head: LinearConfig::new(8, 4).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.pool.forward(relu(self.conv1.forward(x))); // [B, 4, 6, 6]
        let x = relu(self.conv2.forward(x)); //                    [B, 8, 6, 6]
        let [batch, channels, _, _] = x.dims();
        let x = self.gap.forward(x).reshape([batch, channels]); // [B, 8]
        self.head.forward(x) //                                    [B, 4]
    }

    // Replace the random init with fixed numbers, so python.py can use the SAME weights.
    fn with_fixed_weights(mut self, device: &B::Device) -> Self {
        let det = |n: usize, fan_in: usize, phase: f64| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f64 * 2.3 + phase).sin() / (fan_in as f64).sqrt()) as f32)
                .collect()
        };
        self.conv1.weight = param(det(36, 9, 0.1), [4, 1, 3, 3], device);
        self.conv1.bias = Some(param(vec![0.0; 4], [4], device));
        self.conv2.weight = param(det(288, 36, 0.3), [8, 4, 3, 3], device);
        self.conv2.bias = Some(param(vec![0.0; 8], [8], device));
        // PyTorch stores Linear as [out, in]; Burn stores [in, out]. Transpose.
        let w = det(32, 8, 0.5); // [4, 8] in PyTorch's order
        let w_t: Vec<f32> = (0..32).map(|k| w[(k % 4) * 8 + k / 4]).collect(); // -> [8, 4]
        self.head.weight = param(w_t, [8, 4], device);
        self.head.bias = Some(param(vec![0.0; 4], [4], device));
        self
    }
}

fn param<B: Backend, const D: usize>(
    v: Vec<f32>,
    shape: [usize; D],
    device: &B::Device,
) -> Param<Tensor<B, D>> {
    Param::from_tensor(Tensor::from_data(TensorData::new(v, shape), device))
}

// The same classifier with no convolutions: flatten the image, two dense layers.
#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

/// Returns (logits for one image of each class, cnn params, mlp params).
fn build() -> (Vec<f32>, usize, usize) {
    let device = Default::default();
    let mut rng = Lcg(42);
    let images: Vec<f32> = (0..4).flat_map(|c| make_image(c, &mut rng)).collect();
    let x = Tensor::<MyBackend, 4>::from_data(TensorData::new(images, [4, 1, SIZE, SIZE]), &device);

    let model = SmallCnn::<MyBackend>::new(&device).with_fixed_weights(&device);
    let logits = model.forward(x);
    println!("logits shape = {:?}", logits.dims()); // [4, 4]

    let mlp = Mlp::<MyBackend> {
        l1: LinearConfig::new(SIZE * SIZE, 32).init(&device),
        l2: LinearConfig::new(32, 4).init(&device),
    };
    (
        logits.into_data().to_vec().unwrap(),
        model.num_params(),
        mlp.num_params(),
    )
}

fn main() {
    let (logits, cnn, mlp) = build();
    for (class, row) in logits.chunks(4).enumerate() {
        println!("image of class {class}: logits {row:.4?}");
    }
    println!("CNN parameters = {cnn}");
    println!("MLP parameters = {mlp}  (144 -> 32 -> 4)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (logits, cnn, mlp) = build();
        // Reference logits from python.py (same data, same fixed weights).
        let expected = [
            0.000385, 0.000009, -0.000369, -0.000674, 0.007906, 0.007646, 0.005866, 0.002920,
            -0.001834, -0.001361, -0.000618, 0.000247, -0.001481, -0.000899, -0.000138, 0.000650,
        ];
        for (a, b) in logits.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        assert_eq!(cnn, 372);
        assert_eq!(mlp, 4772);
    }
}
