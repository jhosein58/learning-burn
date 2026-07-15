use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type B = NdArray;

fn main() {
    let dev = Default::default();

    let m = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev);
    let v = Tensor::<B, 1>::from_floats([1.0, 1.0], &dev);

    let mt = m.clone().transpose(); // or .t()
    let mv = linalg::matvec(m.clone(), v.clone()); // matrix * vector -> [2]
    let d = v.clone().dot(v.clone()); // scalar dot product
    let op: Tensor<B, 2> = linalg::outer(v.clone(), v); // outer product -> [2, 2]

    println!("transpose =\n{}", mt.to_data());
    println!("matvec    = {}", mv.to_data()); // [3.0, 7.0]
    println!("dot       = {}", d.to_data()); // [2.0]
    println!("outer     =\n{}", op.to_data());
}
