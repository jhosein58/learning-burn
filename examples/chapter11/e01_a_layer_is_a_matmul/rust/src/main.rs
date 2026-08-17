use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
// Chapter 11 — A neural network from scratch.
//
// No nn::Linear, no Module derive — just weight matrices and matmuls, so you can
// see EXACTLY what a "layer" is: an affine map (x @ W) followed by a nonlinearity.
//
//  h  = relu(x @ W1)    hidden layer    (batch=2, 2 features -> 3 hidden units)
//  out = h @ W2         output layer    (3 hidden -> 1 output)
//
type B = NdArray;

fn main() {
    let dev = Default::default();

    // A batch of two examples, each with two features.
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev);

    // Expected targets for the input batch — one target per example, so [2, 1],
    // matching the shape the network outputs.
    let t = Tensor::<B, 2>::from_floats([[1.0], [0.0]], &dev);

    // Fixed weights so every number below is reproducible.
    let w1 = Tensor::<B, 2>::from_floats([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], &dev);

    let w2 = Tensor::<B, 2>::from_floats([[0.7], [0.8], [0.9]], &dev);
    // Forward pass, layer by layer.
    let h_pre = x.clone().matmul(w1.clone()); // pre-activation [2, 3]

    let h = relu(h_pre.clone()); // hidden activations [2, 3]
    let out = h.clone().matmul(w2.clone()); // output  [2, 1]

    // Mean squared error: square the per-example errors, then average them
    // down to a single scalar.
    let diff = out.clone() - t.clone(); // [2, 1]
    let sq_err = diff.clone() * diff.clone(); // [2, 1]
    let loss = sq_err.clone().mean(); // scalar

    println!("xshape  = {:?}", x.dims()); // [2, 2]
    println!("W1shape = {:?}", w1.dims()); // [2, 3]
    println!("W2shape = {:?}", w2.dims()); // [3, 1]
    println!("h_pre   = {}", h_pre.to_data()); // [[0.9, 1.2, 1.5], [1.9, 2.6, 3.3]]
    println!("h       = {}", h.to_data()); // same (all positive, so relu is identity)
    println!("out     = {}", out.to_data()); // [[2.94], [6.38]]
    println!("t  = {}", t.to_data()); // [[1.0], [0.0]]
    println!("sq_err  = {}", sq_err.to_data()); // [[3.7636], [40.7044]]
    println!("loss    = {}", loss.to_data()); // 22.234 (scalar MSE)
}
