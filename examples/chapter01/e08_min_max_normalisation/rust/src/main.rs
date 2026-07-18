use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let input = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);

    let min = input.clone().min();
    let max = input.clone().max();
    let input = (input.clone() - min.clone()).div(max - min);

    println!("{}", input.to_data());
    // [0.0, 0.33333334, 0.6666667, 1.0]
}
