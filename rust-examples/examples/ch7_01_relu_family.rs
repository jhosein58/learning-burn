use burn::backend::NdArray;
use burn::tensor::{Tensor, activation};

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let x = Tensor::<Backend, 1>::from_floats([-2.0, -0.5, 0.0, 1.0, 3.0], &device);

    let relu = activation::relu(x.clone());
    let leaky = activation::leaky_relu(x.clone(), 0.1);
    let gelu = activation::gelu(x.clone());

    println!("relu       = {}", relu.to_data()); // [0, 0, 0, 1, 3]
    println!("leaky_relu = {}", leaky.to_data()); // [-0.2, -0.05, 0, 1, 3]
    println!("gelu       = {}", gelu.to_data()); // smooth ~[-0.05, -0.15, 0, 0.84, 2.996]
}
