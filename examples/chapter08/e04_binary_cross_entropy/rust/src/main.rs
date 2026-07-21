use burn::backend::NdArray;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::tensor::{Int, Tensor};

type B = NdArray;

fn main() {
    let dev = Default::default();

    let logits = Tensor::<B, 1>::from_floats([2.0, -1.5, 0.3], &dev);
    let targets = Tensor::<B, 1, Int>::from_ints([1, 0, 1], &dev);

    let loss = BinaryCrossEntropyLossConfig::new()
        .with_logits(true) // inputs are logits; sigmoid applied inside
        .init(&dev)
        .forward(logits, targets);

    println!("loss = {}", loss.to_data()); // ~0.294
}
