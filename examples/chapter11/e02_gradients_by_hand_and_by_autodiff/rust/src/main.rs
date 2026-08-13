use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
// Chapter 11 — Backpropagation, demystified.
//
// This is the example everyone needs: we compute the gradients TWICE.
//      1. By hand, using the backprop formulas (the chain rule, one layer at a time).
//      2. With autodiff (loss.backward()).
// They come out identical. Once you see that, backprop stops being magic.
//
// Network (batch=2, no bias): out = relu(x @ W1) @ W2,  loss = mean((out - t)^2).
//
// The backprop recipe (walking backwards through the graph):
//      dout= (2/N) * (out - t)        # d loss / d out
//      dW2= hᵀ @ dout                 # gradient for layer 2   [N = batch size]
//      dh= dout @ W2ᵀ                 # push error into hidden activations
//      dh_pre= dh * (h_pre > 0)      # back through relu: pass where input was > 0
//      dW1= xᵀ @ dh_pre               # gradient for layer 1
type Backend = Autodiff<NdArray>;

fn main() {
    let device = Default::default();

    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let t = Tensor::<Backend, 2>::from_floats([[1.0], [0.0]], &device);
    let n = x.dims()[0] as f64; // batch size = 2

    let w1 = Tensor::<Backend, 2>::from_floats([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], &device)
        .require_grad();
    let w2 = Tensor::<Backend, 2>::from_floats([[0.7], [0.8], [0.9]], &device).require_grad();

    // ---- forward ----
    let h_pre = x.clone().matmul(w1.clone()); // [2, 3]
    let h = relu(h_pre.clone()); // [2, 3]
    let out = h.clone().matmul(w2.clone()); // [2, 1]
    let diff = out - t; // [2, 1]
    let loss = (diff.clone() * diff.clone()).mean(); // scalar MSE

    // ---- (1) gradients BY HAND ----
    let dout = diff.mul_scalar(2.0 / n); // [2, 1]
    let manual_gw2 = h.clone().transpose().matmul(dout.clone()); // [3, 1]
    let dh = dout.matmul(w2.clone().transpose()); // [2, 3]

    // back through relu: zero out positions where the pre-activation was <= 0
    let dh_pre = dh.mask_fill(h_pre.lower_equal_elem(0.0), 0.0); // [2, 3]
    let manual_gw1 = x.transpose().matmul(dh_pre); // [2, 3]

    // ---- (2) gradients from AUTODIFF ----
    let grads = loss.backward();
    let auto_gw1 = w1.grad(&grads).unwrap();
    let auto_gw2 = w2.grad(&grads).unwrap();

    println!("loss = {}", loss.to_data()); // [22.234]
    println!("\n-- gradient for W2 --");
    println!("manual =\n{}", manual_gw2.to_data()); // [[13.868], [18.916], [23.964]]
    println!("auto   =\n{}", auto_gw2.to_data()); // identical
    println!("\n-- gradient for W1 --");
    println!("manual =\n{}", manual_gw1.to_data()); // [[14.756, 16.864, 18.972], [20.58, 23.52, 26.46]]
    println!("auto   =\n{}", auto_gw1.to_data()); // identical
}
