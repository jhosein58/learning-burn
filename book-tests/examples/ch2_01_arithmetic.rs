use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let b = Tensor::<Backend, 1>::from_floats([10.0, 20.0, 30.0], &device);

    // Operators and their named-method equivalents; reused tensors are cloned.
    let sum = a.clone() + b.clone(); // or a.clone().add(b.clone())
    let prod = a.clone() * b.clone(); // element-wise, NOT matmul
    let scaled = a.clone().mul_scalar(2.0);
    let neg = -a;

    println!("sum    = {}", sum.to_data()); // [11.0, 22.0, 33.0]
    println!("prod   = {}", prod.to_data()); // [10.0, 40.0, 90.0]
    println!("scaled = {}", scaled.to_data()); // [2.0, 4.0, 6.0]
    println!("neg    = {}", neg.to_data()); // [-1.0, -2.0, -3.0]
}
