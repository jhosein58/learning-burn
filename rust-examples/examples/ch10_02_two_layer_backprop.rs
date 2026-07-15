use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use burn::tensor::activation::relu;

// The heart of backpropagation: gradients flow *through a hidden layer* back to
// the first weight matrix. We build a tiny 2-layer net by hand with fixed
// weights so every number is reproducible:
//
//   h_pre = x @ W1      (pre-activation)
//   h     = relu(h_pre) (hidden activations)
//   out   = h @ W2
//   loss  = (out - target)^2
//
// PyTorch counterpart: same tensors with requires_grad=True, then loss.backward().
// Because relu'(z) = 1 for z > 0, the gradient passes cleanly to both layers here.
type Backend = Autodiff<NdArray>;

fn main() {
    let device = Default::default();

    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0]], &device);
    let target = Tensor::<Backend, 2>::from_floats([[1.0]], &device);

    // Fixed weights so the gradients are deterministic and hand-checkable.
    let w1 = Tensor::<Backend, 2>::from_floats([[0.1, 0.2], [0.3, 0.4]], &device).require_grad();
    let w2 = Tensor::<Backend, 2>::from_floats([[0.5], [0.6]], &device).require_grad();

    // Forward pass.
    let h_pre = x.matmul(w1.clone()); // [[0.7, 1.0]]
    let h = relu(h_pre); // [[0.7, 1.0]] (both positive)
    let out = h.matmul(w2.clone()); // [[0.95]]
    let diff = out.clone() - target;
    let loss = (diff.clone() * diff).sum(); // (0.95 - 1.0)^2 = 0.0025

    // Backward pass: one call fills in dL/dW1 AND dL/dW2.
    let grads = loss.backward();
    let g_w1 = w1.grad(&grads).unwrap();
    let g_w2 = w2.grad(&grads).unwrap();

    println!("out  = {}", out.to_data()); // [[0.95]]
    println!("loss = {}", loss.to_data()); // [0.0025]
    // Gradient that reached the SECOND layer:
    println!("gW2  =\n{}", g_w2.to_data()); // [[-0.07], [-0.10]]
    // Gradient that propagated all the way back to the FIRST layer:
    println!("gW1  =\n{}", g_w1.to_data()); // [[-0.05, -0.06], [-0.10, -0.12]]
}
