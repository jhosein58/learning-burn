use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

// Toward ordinary least squares: build the pieces of the normal equations,
// X^T X and X^T y. The final `solve`/`inverse` is exactly what Burn is still
// missing (issue #1538: linalg solve / inv / QR / SVD).
fn main() {
    let device = Default::default();

    // design matrix X: [n_samples, n_features]
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], &device);
    let y = Tensor::<Backend, 2>::from_floats([[6.0], [0.0], [0.0]], &device); // [n, 1]

    let xt = x.clone().transpose();
    let gram = xt.clone().matmul(x.clone()); // X^T X -> [2, 2]
    let xty = xt.matmul(y.clone()); // X^T y -> [2, 1]

    println!("X^T X =\n{}", gram.to_data()); // [[3, 6], [6, 14]]
    println!("X^T y =\n{}", xty.to_data()); // [[6], [6]]
    // w = (X^T X)^-1 X^T y  <-- needs linalg::solve / inv (issue #1538)
}
