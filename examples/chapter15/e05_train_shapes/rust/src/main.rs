use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};

// Training needs gradients, so wrap the CPU backend in Autodiff (Chapter 5).
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

/// Returns (mean loss per epoch, correct test predictions out of 80).
fn build() -> (Vec<f32>, usize) {
    let device = Default::default();
    let train = shapes_dataset(40, 42); // 160 images
    let test = shapes_dataset(20, 7); //   80 images, a different seed

    // No shuffle here, so the order matches python.py exactly (shuffling: Chapter 17).
    let loader =
        DataLoaderBuilder::<MyBackend, ShapeItem, ShapeBatch<MyBackend>>::new(ShapeBatcher)
            .batch_size(16)
            .build(train);

    let mut model = SmallCnn::<MyBackend>::new(&device).with_fixed_weights(&device);
    // Burn's Adam defaults to epsilon 1e-5; PyTorch's to 1e-8. Match PyTorch.
    let mut optim = AdamConfig::new().with_epsilon(1e-8).init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let lr = 0.02;

    let mut history = Vec::new();
    for epoch in 1..=15 {
        let (mut total, mut batches) = (0.0, 0);
        for batch in loader.iter() {
            let logits = model.forward(batch.images);
            let loss = loss_fn.forward(logits, batch.targets);
            total += loss.clone().into_scalar();
            batches += 1;

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
        }
        let mean = total / batches as f32;
        println!("epoch {epoch:>2}  loss {mean:.4}");
        history.push(mean);
    }

    let test_batch = ShapeBatcher.batch(test.iter().collect(), &device);
    (history, accuracy(&model, test_batch))
}

fn main() {
    let (_, correct) = build();
    println!("test accuracy = {correct}/80");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (history, correct) = build();
        // Mean loss per epoch from python.py: same data, same order, same init.
        let expected = [
            1.344356, 1.104616, 0.853722, 0.715782, 0.585857, 0.470269, 0.384723, 0.276252,
            0.130639, 0.056581, 0.031400, 0.028396, 0.021925, 0.020555, 0.009121,
        ];
        for (a, b) in history.iter().zip(expected) {
            assert!((a - b).abs() < 2e-3, "{a} vs {b}");
        }
        assert_eq!(correct, 80);
    }
}
