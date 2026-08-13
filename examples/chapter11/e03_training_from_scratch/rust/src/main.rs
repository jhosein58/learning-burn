use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Distribution, Tensor};

// Chapter 11 — Training a network entirely from scratch.
//
// No nn::Linear, no Module, no Optimizer. Just four weight tensors and the
// four-beat loop you now understand: forward -> loss -> backward -> update.
// We let autodiff compute the gradients (ch11_02 showed those formulas are real),
// but we apply the SGD update ourselves: param <- param - lr * grad.
//
// Task: learn XOR, which needs a hidden layer + nonlinearity.

type Backend = Autodiff<NdArray>;

fn main() {
    let device = Default::default();

    let x = Tensor::<Backend, 2>::from_floats(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        &device,
    );
    let y = Tensor::<Backend, 2>::from_floats([[0.0], [1.0], [1.0], [0.0]], &device);

    let hidden = 8;

    // Random init breaks symmetry so hidden units can learn different features.
    // Biases start at zero. require_grad() marks each as a trainable leaf.
    let mut w1 =
        Tensor::<Backend, 2>::random([2, hidden], Distribution::Uniform(-1.0, 1.0), &device)
            .require_grad();
    let mut b1 = Tensor::<Backend, 2>::zeros([1, hidden], &device).require_grad();
    let mut w2 =
        Tensor::<Backend, 2>::random([hidden, 1], Distribution::Uniform(-1.0, 1.0), &device)
            .require_grad();
    let mut b2 = Tensor::<Backend, 2>::zeros([1, 1], &device).require_grad();

    let lr = 0.1;

    for epoch in 0..40000 {
        // Forward: h = tanh(x @ W1 + b1); out = h @ W2 + b2
        let h = (x.clone().matmul(w1.clone()) + b1.clone()).tanh();
        let out = h.matmul(w2.clone()) + b2.clone();
        let diff = out - y.clone();
        let loss = (diff.clone() * diff).mean();

        if epoch % 8000 == 0 {
            println!("epoch {:>5}  loss = {}", epoch, loss.to_data());
        }

        // Backward: one call fills the gradient for every leaf.
        let grads = loss.backward();
        let g_w1 = w1.grad(&grads).unwrap();
        let g_b1 = b1.grad(&grads).unwrap();
        let g_w2 = w2.grad(&grads).unwrap();
        let g_b2 = b2.grad(&grads).unwrap();

        // Manual SGD step. inner() drops down to the plain backend to do the
        // arithmetic; from_inner(...).require_grad() lifts the result back into
        // a fresh trainable leaf for the next iteration.
        w1 = Tensor::from_inner(w1.inner() - g_w1.mul_scalar(lr)).require_grad();
        b1 = Tensor::from_inner(b1.inner() - g_b1.mul_scalar(lr)).require_grad();
        w2 = Tensor::from_inner(w2.inner() - g_w2.mul_scalar(lr)).require_grad();
        b2 = Tensor::from_inner(b2.inner() - g_b2.mul_scalar(lr)).require_grad();
    }

    let h = (x.clone().matmul(w1.clone()) + b1.clone()).tanh();
    let preds = h.matmul(w2.clone()) + b2.clone();
    println!(
        "\nfinal predictions (target 0, 1, 1, 0):\n{}",
        preds.to_data()
    );
    // Random init means a rare run gets stuck; if outputs aren't ~0,1,1,0, rerun.
}
