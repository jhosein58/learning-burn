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
// PyTorch counterpart (also from scratch):
//  h   = torch.relu(x @ W1)
//  out = h @ W2
type Backend = NdArray;

fn main() {
    let device = Default::default();

    // A batch of two examples, each with two features.
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

    // Fixed weights so every number below is reproducible.
    let w1 = Tensor::<Backend, 2>::from_floats([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], &device);

    let w2 = Tensor::<Backend, 2>::from_floats([[0.7], [0.8], [0.9]], &device);
    // Forward pass, layer by layer.
    let h_pre = x.clone().matmul(w1.clone()); // pre-activation [2, 3]

    let h = relu(h_pre.clone()); // hidden activations [2, 3]
    let out = h.clone().matmul(w2.clone()); // output  [2, 1]

    println!("xshape  = {:?}", x.dims()); // [2, 2]
    println!("W1shape = {:?}", w1.dims()); // [2, 3]
    println!("W2shape = {:?}", w2.dims()); // [3, 1]
    println!("h_pre   =\n{}", h_pre.to_data()); // [[0.9, 1.2, 1.5], [1.9, 2.6, 3.3]]
    println!("h       =\n{}", h.to_data()); // same (all positive, so relu is identity)
    println!("out     =\n{}", out.to_data()); // [[2.94], [6.38]]
}
