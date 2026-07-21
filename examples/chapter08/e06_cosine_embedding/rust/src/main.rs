use burn::backend::NdArray;
use burn::nn::loss::CosineEmbeddingLossConfig;
use burn::tensor::{Int, Tensor};

type B = NdArray;

fn main() {
    let dev = Default::default();

    // two batches of embeddings; +1 = "should match", -1 = "should differ"
    let a = Tensor::<B, 2>::from_floats([[1.0, 0.0], [1.0, 0.0]], &dev);
    let b = Tensor::<B, 2>::from_floats([[0.9, 0.1], [-1.0, 0.0]], &dev);
    let y = Tensor::<B, 1, Int>::from_ints([1, -1], &dev);

    let loss = CosineEmbeddingLossConfig::new().init().forward(a, b, y);

    println!("loss = {}", loss.to_data()); // ~0.003
}
