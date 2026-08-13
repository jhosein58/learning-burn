use burn::backend::NdArray;
use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::Tensor;

type B = NdArray;

fn main() {
    let dev = Default::default();

    let pred = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev);
    let target = Tensor::<B, 2>::from_floats([[2.0, 1.0], [3.0, 2.0]], &dev);

    let mse = MseLoss::new();
    let loss = mse.forward(pred.clone(), target.clone(), Reduction::Mean);

    println!("MSE = {}", loss.to_data()); // mean of [1, 1, 0, 4] = 1.5
}
