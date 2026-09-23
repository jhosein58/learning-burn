use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::module::{AutodiffModule, Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};

type MyBackend = Autodiff<NdArray>;

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

// --- feeding the model: Dataset -> Batcher -> DataLoader (Chapter 17 goes deeper) ---
#[derive(Clone, Debug)]
struct ShapeItem {
    pixels: Vec<f32>,
    label: usize,
}

// Class order cycles 0,1,2,3,0,1,... so every batch of 16 is balanced.
fn shapes_dataset(per_class: usize, seed: u32) -> InMemDataset<ShapeItem> {
    let mut rng = Lcg(seed);
    let items = (0..per_class * 4)
        .map(|i| ShapeItem {
            pixels: make_image(i % 4, &mut rng),
            label: i % 4,
        })
        .collect();
    InMemDataset::new(items)
}

#[derive(Clone, Debug)]
struct ShapeBatch<B: Backend> {
    images: Tensor<B, 4>,       // [batch, 1, 12, 12]
    targets: Tensor<B, 1, Int>, // [batch]
}

#[derive(Clone, Default)]
struct ShapeBatcher;

impl<B: Backend> Batcher<B, ShapeItem, ShapeBatch<B>> for ShapeBatcher {
    fn batch(&self, items: Vec<ShapeItem>, device: &B::Device) -> ShapeBatch<B> {
        let n = items.len();
        let pixels: Vec<f32> = items.iter().flat_map(|it| it.pixels.clone()).collect();
        let labels: Vec<i64> = items.iter().map(|it| it.label as i64).collect();
        ShapeBatch {
            images: Tensor::from_data(TensorData::new(pixels, [n, 1, SIZE, SIZE]), device),
            targets: Tensor::from_data(TensorData::new(labels, [n]), device),
        }
    }
}

fn accuracy<B: Backend>(model: &SmallCnn<B>, batch: ShapeBatch<B>) -> usize {
    let n = batch.targets.dims()[0];
    let predicted = model.forward(batch.images).argmax(1).reshape([n]);
    predicted
        .equal(batch.targets)
        .int()
        .sum()
        .into_scalar()
        .elem::<i64>() as usize
}

// Example 15.5's training run, unchanged: 15 epochs of Adam on the shapes.
fn train() -> SmallCnn<MyBackend> {
    let device = &Default::default();
    let loader =
        DataLoaderBuilder::<MyBackend, ShapeItem, ShapeBatch<MyBackend>>::new(ShapeBatcher)
            .batch_size(16)
            .build(shapes_dataset(40, 42));
    let mut model = SmallCnn::<MyBackend>::new(device).with_fixed_weights(device);
    let mut optim = AdamConfig::new().with_epsilon(1e-8).init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    for _ in 0..15 {
        for batch in loader.iter() {
            let loss = loss_fn.forward(model.forward(batch.images), batch.targets);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(0.02, model, grads);
        }
    }
    model
}

// Chapter 14's symmetric int8 scheme, applied to a whole weight tensor at once.
// Returns the dequantised weights (what inference will actually use) and the scale.
fn int8_round_trip<B: Backend, const D: usize>(w: Tensor<B, D>) -> (Tensor<B, D>, f32) {
    let scale = w.clone().abs().max().into_scalar().elem::<f32>() / 127.0;
    let codes = w.div_scalar(scale).round().clamp(-127.0, 127.0); // whole numbers, one byte each
    (codes.mul_scalar(scale), scale)
}

/// Returns (conv1 filters, f32 correct, int8 correct, f32 bytes, int8 bytes).
fn build() -> (Vec<f32>, usize, usize, usize, usize) {
    let device = Default::default();
    // .valid() drops the autodiff wrapper: inference runs on plain NdArray.
    let model = train().valid();
    let test = shapes_dataset(20, 7);
    let batch = || ShapeBatcher.batch(test.iter().collect(), &device);

    let filters: Vec<f32> = model.conv1.weight.val().into_data().to_vec().unwrap();
    let f32_correct = accuracy(&model, batch());

    // Quantise the three weight tensors. Biases (16 numbers) stay f32.
    let mut q = model.clone();
    let (w1, s1) = int8_round_trip(q.conv1.weight.val());
    let (w2, s2) = int8_round_trip(q.conv2.weight.val());
    let (w3, s3) = int8_round_trip(q.head.weight.val());
    println!("scales         = [{s1:.5}, {s2:.5}, {s3:.5}]");
    q.conv1.weight = Param::from_tensor(w1);
    q.conv2.weight = Param::from_tensor(w2);
    q.head.weight = Param::from_tensor(w3);
    let int8_correct = accuracy(&q, batch());

    let weights = 36 + 288 + 32; // conv1 + conv2 + head
    let biases = 4 + 8 + 4;
    let f32_bytes = (weights + biases) * 4;
    let int8_bytes = weights + 3 * 4 + biases * 4; // codes + 3 scales + f32 biases
    (filters, f32_correct, int8_correct, f32_bytes, int8_bytes)
}

fn main() {
    let (filters, f32_correct, int8_correct, f32_bytes, int8_bytes) = build();
    for (k, f) in filters.chunks(9).enumerate() {
        println!(
            "conv1 filter {k}: {:.2?} {:.2?} {:.2?}",
            &f[0..3],
            &f[3..6],
            &f[6..9]
        );
    }
    println!("f32  model: {f32_correct}/80 correct, {f32_bytes} bytes");
    println!("int8 model: {int8_correct}/80 correct, {int8_bytes} bytes");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (filters, f32_correct, int8_correct, f32_bytes, int8_bytes) = build();
        // conv1 after training, from python.py.
        let expected = [
            -0.422270, 0.930256, -0.156822, 0.394610, 0.132003, -0.237828, 1.001078, -0.483036,
            -0.079344, 0.456176, -0.339555, 0.453012, 0.761879, -1.099708, 0.817598, 0.453400,
            -0.468780, 0.602896, -0.596102, 0.489929, 0.240630, -1.281593, 0.889172, 0.521594,
            -0.261456, 1.098831, -0.865395, -1.319253, 1.045184, -0.067182, 0.521430, 0.890422,
            -1.672505, -0.141084, 1.779191, -1.678253,
        ];
        for (a, b) in filters.iter().zip(expected) {
            assert!((a - b).abs() < 2e-3, "{a} vs {b}");
        }
        assert_eq!((f32_correct, int8_correct), (80, 80));
        assert_eq!((f32_bytes, int8_bytes), (1488, 432));
    }
}
