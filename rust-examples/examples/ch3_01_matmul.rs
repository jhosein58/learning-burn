use burn::backend::NdArray;
use burn::tensor::Tensor;

type B = NdArray;

fn main() {
    let dev = Default::default();

    // [m, k] times [k, n] -> [m, n]; the inner dimensions must match.
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &dev); // [2, 3]
    let b = Tensor::<B, 2>::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], &dev); // [3, 2]

    // Matrix product (element-wise `*` would NOT be matmul).
    let c = a.matmul(b); // [2, 2]

    println!("{}", c.to_data());
}
