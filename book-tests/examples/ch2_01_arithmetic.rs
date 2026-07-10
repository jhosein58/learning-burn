use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let b = Tensor::<Backend, 1>::from_floats([10.0, 20.0, 30.0], &device);

    // Operators and their named-method equivalents; reused tensors are cloned.
    let s = a.clone() + b.clone(); // or a.clone().add(b.clone())
    let p = a.clone() * b.clone(); // element-wise, NOT matmul
    let sc = a.clone().mul_scalar(2.0); // or a.clone().mul_scalar(2.0)
    let neg = -a;

    println!("sum    = {}", s.to_data()); // [11.0, 22.0, 33.0]
    println!("prod   = {}", p.to_data()); // [10.0, 40.0, 90.0]
    println!("scaled = {}", sc.to_data()); // [2.0, 4.0, 6.0]
    println!("neg    = {}", neg.to_data()); // [-1.0, -2.0, -3.0]
}
