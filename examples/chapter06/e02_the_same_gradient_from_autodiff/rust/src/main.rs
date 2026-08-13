use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type B = Autodiff<NdArray>;

// Instead of deriving the MSE gradient by hand (ch6_01), let autodiff compute it.
// At w = 0 the gradient equals -(2/n) X^T y, which is [-12, -35] for this data.
fn main() {
    let dev = Default::default();

    let x = Tensor::<B, 2>::from_floats(
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]],
        &dev,
    );

    let y = Tensor::<B, 2>::from_floats([[3.0], [5.0], [7.0], [9.0]], &dev);

    let w = Tensor::<B, 2>::zeros([2, 1], &dev).require_grad();

    let pred = x.matmul(w.clone());
    let diff = pred - y;
    let loss = (diff.clone() * diff).mean(); // MSE

    let grads = loss.backward();
    let gw = w.grad(&grads).unwrap();

    println!("loss = {}", loss.to_data()); // mean of [9, 25, 49, 81] = 41
    println!("grad = {}", gw.to_data()); // [-12.0, -35.0] (matches the hand formula)
}
