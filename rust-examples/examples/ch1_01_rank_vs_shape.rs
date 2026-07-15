use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let floats = [1.0, 2.0, 3.0, 4.0, 5.0];
    let device = Default::default();

    // CORRECT: a 1-dimensional tensor that happens to hold 5 elements
    let tensor = Tensor::<Backend, 1>::from_floats(floats, &device);

    println!("tensor = {}", tensor);

    // WRONG: this would ask for a 5-dimensional tensor and will not compile as intended
    // let tensor = Tensor::<Backend, 5>::from_floats(floats, &device);
}
