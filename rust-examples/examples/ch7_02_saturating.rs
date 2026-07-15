use burn::backend::NdArray;
use burn::tensor::{Tensor, activation};

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let x = Tensor::<Backend, 1>::from_floats([-1.0, 0.0, 1.0], &device);

    // Saturating / output units: sigmoid squashes to (0, 1), tanh to (-1, 1).
    let sigmoid = activation::sigmoid(x.clone());
    let tanh = x.clone().tanh();

    println!("sigmoid = {}", sigmoid.to_data()); // [0.26894, 0.5, 0.73106]
    println!("tanh    = {}", tanh.to_data()); // [-0.76159, 0.0, 0.76159]
}
