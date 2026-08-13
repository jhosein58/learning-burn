use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn::tensor::activation::softmax;

// Chapter 12 — Causal (masked) self-attention.
//
// In a language model, token i must not attend to tokens that come AFTER it.
// We enforce that with an additive mask: add a huge negative number to every
// "future" score (positions j > i) BEFORE the softmax, so those weights vanish.
//
// The mask is lower-triangular-friendly:
//      [[ 0, -inf],
//       [ 0,  0 ]]
// PyTorch does the same with masked_fill(mask, float('-inf')).
//
//
type Backend = NdArray;

fn main() {
    let device = Default::default();

    let q = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let k = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let v = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

    let d_k = q.dims()[1] as f64;
    let scale = 1.0 / d_k.sqrt();

    let scores = q.matmul(k.transpose()).mul_scalar(scale);

    // Additive causal mask: 0 where attention is allowed, -1e9 where it's forbidden.
    let mask = Tensor::<Backend, 2>::from_floats([[0.0, -1.0e9], [0.0, 0.0]], &device);
    let masked = scores + mask;

    let weights = softmax(masked, 1);
    let out = weights.clone().matmul(v);

    // Row 0 can only see position 0, so it copies V's first row exactly.
    println!("masked weights =\n{}", weights.to_data()); // [[1, 0], [0.3302, 0.6698]]
    println!("output         =\n{}", out.to_data()); // [[1, 2], [2.3395, 3.3395]]
}
