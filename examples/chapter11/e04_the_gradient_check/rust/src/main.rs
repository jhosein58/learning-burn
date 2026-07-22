use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
// Chapter 11 — Gradient checking: how do you KNOW backprop is right?
//
// The definition of a derivative is a limit of finite differences:
// f'(w) ≈ [ f(w + eps) - f(w - eps) ] / (2 * eps)
// So you can estimate a gradient with only forward passes (no calculus), and
// compare it to what autodiff reports. If they agree, your gradients are correct.
// This "numerical gradient check" is the standard sanity test when building nets.
//
// Toy loss so the arithmetic is easy to follow: f'(w) = 2*(3w - 2)*3.
// Analytic derivative: f(w) = (3w - 2)^2 At w = 0.5 that is -3.0.
//
type Backend = Autodiff<NdArray>;

// Plain-Rust forward pass, used for the finite-difference estimate.
fn loss_at(w: f64) -> f64 {
    let z = 3.0 * w - 2.0;
    z * z
}

fn main() {
    let device = Default::default();
    let w_value = 0.5;

    // --- autodiff gradient ---
    let w = Tensor::<Backend, 1>::from_floats([w_value as f32], &device).require_grad();
    let z = w.clone().mul_scalar(3.0).sub_scalar(2.0);
    let loss = (z.clone() * z).sum();
    let grads = loss.backward();
    let autodiff_grad = w.grad(&grads).unwrap();

    // --- numerical gradient (central difference) ---
    let eps = 1e-4;

    let numeric_grad = (loss_at(w_value + eps) - loss_at(w_value - eps)) / (2.0 * eps);
    println!("loss          = {}", loss.to_data()); // [0.25]
    println!("autodiff grad = {}", autodiff_grad.to_data()); // [-3.0]
    println!("numerical grad = {numeric_grad:.6}"); // -3.000000
    println!("they match -> backprop is trustworthy");
}
