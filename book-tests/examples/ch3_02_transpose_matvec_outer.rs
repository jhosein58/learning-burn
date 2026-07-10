use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let m = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let v = Tensor::<Backend, 1>::from_floats([1.0, 1.0], &device);

    // transpose swaps the last two dims.
    let mt = m.clone().transpose(); // [[1, 3], [2, 4]]

    // matrix * vector, built from matmul: reshape v to a column, then flatten back.
    let mv = m.clone().matmul(v.clone().reshape([2, 1])).reshape([2]); // [3, 7]

    // dot product = element-wise product summed.
    let dot = (v.clone() * v.clone()).sum(); // 2

    // outer product = [n,1] * [1,m] via broadcasting.
    let outer = v.clone().reshape([2, 1]) * v.clone().reshape([1, 2]); // [[1,1],[1,1]]

    println!("transpose =\n{}", mt.to_data());
    println!("matvec    = {}", mv.to_data()); // [3.0, 7.0]
    println!("dot       = {}", dot.to_data()); // [2.0]
    println!("outer     =\n{}", outer.to_data());
}
