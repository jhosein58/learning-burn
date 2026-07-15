use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

// Fit y = b + slope * x by gradient descent, computing the MSE gradient by hand.
// Data is generated from y = 2x + 1, so w should converge to [1, 2].
fn main() {
    let device = Default::default();

    // Design matrix with a bias column: rows are [1, x_i].
    let x = Tensor::<Backend, 2>::from_floats(
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]],
        &device,
    );
    let y = Tensor::<Backend, 2>::from_floats([[3.0], [5.0], [7.0], [9.0]], &device); // y = 2x + 1

    let mut w = Tensor::<Backend, 2>::zeros([2, 1], &device); // [bias, slope]
    let n = 4.0;
    let lr = 0.05;

    for _ in 0..3000 {
        let pred = x.clone().matmul(w.clone()); // Xw
        let err = pred - y.clone(); // Xw - y
        // gradient of MSE: (2/n) X^T (Xw - y)
        let grad = x.clone().transpose().matmul(err).mul_scalar(2.0 / n);
        w = w - grad.mul_scalar(lr);
    }

    println!("w = {}", w.to_data()); // ~[1.0, 2.0]  (bias, slope)
}
