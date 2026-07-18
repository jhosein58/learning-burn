use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor};

type Backend = NdArray;

fn main() {
    let device = Default::default();

    // uniform on [0, 1)
    let u = Tensor::<Backend, 2>::random([2, 3], Distribution::Default, &device);

    // standard normal (mean 0, std 1)
    let n = Tensor::<Backend, 2>::random([2, 3], Distribution::Normal(0.0, 1.0), &device);

    println!("uniform = {}", u);
    println!("normal = {}", n);
}
