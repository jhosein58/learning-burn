use burn::backend::NdArray;
use burn::nn::loss::{HuberLossConfig, Reduction};
use burn::tensor::Tensor;

type B = NdArray;

fn main() {
    let dev = Default::default();

    let pred = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev);
    let target = Tensor::<B, 2>::from_floats([[2.0, 1.0], [3.0, 2.0]], &dev);

    let loss = HuberLossConfig::new(1.0) // delta = 1.0
        .init()
        .forward(pred, target, Reduction::Mean);

    println!("loss = {}", loss.to_data()); // 0.625
}
