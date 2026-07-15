use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type B = NdArray;

fn main() {
    let dev = Default::default();

    let x = Tensor::<B, 1>::from_floats([3.0, 4.0], &dev);
    let n2 = linalg::l2_norm(x.clone(), 0); // 5.0

    let unit = linalg::vector_normalize(x, 2.0, 0, 1e-12); // [0.6, 0.8]

    // `det` operates on a BATCH of square matrices, so the input rank must be
    // at least 3: [1, 2, 2] is a batch of one 2x2 matrix. A plain [2, 2] panics.
    // The two extra generics are fully determined by the input rank D:
    //   D1 = D - 1   (rank of the pivot vector used internally)
    //   D2 = D - 2   (rank of the result: the two matrix axes are consumed)
    let m = Tensor::<B, 3>::from_floats([[[4.0, 3.0], [6.0, 3.0]]], &dev);
    let d = linalg::det::<B, 3, 2, 1>(m); // -6.0

    println!("n2   = {}", n2.to_data());
    println!("d    = {}", d.to_data());
    println!("unit = {}", unit.to_data());
}
