use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

// Burn 0.16 has no dedicated `linalg` module, so we build the norms from core
// tensor ops. (SVD / QR / solve are still missing upstream — issue #1538.)
fn main() {
    let device = Default::default();

    let x = Tensor::<Backend, 1>::from_floats([3.0, 4.0], &device);

    // L2 norm = sqrt(sum(x^2))
    let l2 = (x.clone() * x.clone()).sum().sqrt(); // 5.0
    // L1 norm = sum(|x|)
    let l1 = x.clone().abs().sum(); // 7.0
    // unit-normalize: divide by the L2 norm (broadcasts)
    let unit = x.clone() / l2.clone(); // [0.6, 0.8]

    println!("l2   = {}", l2.to_data()); // [5.0]
    println!("l1   = {}", l1.to_data()); // [7.0]
    println!("unit = {}", unit.to_data()); // [0.6, 0.8]
}
