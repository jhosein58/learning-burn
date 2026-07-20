use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    // design matrix X: [n_samples, n_features]
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], &device);
    let y = Tensor::<Backend, 2>::from_floats([[6.0], [0.0], [0.0]], &device);

    let xt = x.clone().transpose();
    let gram = xt.clone().matmul(x.clone()); // X^T X
    let xty = xt.matmul(y.clone()); // X^T y
    // let w = linalg::solve(gram, xty); // <-- needs #1538

    println!("X^T X =\n{}", gram.to_data());
    println!("X^T y =\n{}", xty.to_data());
}
