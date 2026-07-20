use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

// Autodiff is a backend *decorator*: wrap any backend to get gradients.
type Backend = Autodiff<NdArray>;

fn main() {
    let device = Default::default();

    // f(x) = sum(x^2). By hand, df/dx = 2x.
    let x = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();
    let f = (x.clone() * x.clone()).sum(); // 1 + 4 + 9 = 14

    // One call computes the gradient for us.
    let grads = f.backward();
    let dx = x.grad(&grads).unwrap(); // [2, 4, 6] == 2x

    println!("f  = {}", f.to_data()); // [14.0]
    println!("dx = {}", dx.to_data()); // [2.0, 4.0, 6.0]
}
