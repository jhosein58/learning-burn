use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    // [m, k] times [k, n] -> [m, n]; the inner dimensions must match.
    let a = Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device); // [2, 3]
    let b = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], &device); // [3, 2]

    // Matrix product (element-wise `*` would NOT be matmul).
    let c = a.matmul(b); // [2, 2]

    println!("{}", c.to_data());
    // [[4.0, 5.0], [10.0, 11.0]]
}
