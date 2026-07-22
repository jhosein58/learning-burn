use burn::backend::NdArray;
use burn::tensor::activation::softmax;
use burn::tensor::Tensor;
// Chapter 12 — Attention from scratch.
//
// The one equation behind every transformer:
//
//      Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V
//
// Read it as: each query row scores itself against every key (dot product),
// scale down so the softmax stays well-behaved, turn scores into weights that
// sum to 1, then take the weighted average of the value rows.
//
// PyTorch counterpart:
//      w = torch.softmax(Q @ K.T / d_k**0.5, dim=1); out = w @ V
//
//
type Backend = NdArray;

fn main() {
    let device = Default::default();

    // seq_len = 2, d_k = 2. Q and K are the identity so the scores are easy to read.
    let q = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let k = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let v = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

    let d_k = q.dims()[1] as f64;
    let scale = 1.0 / d_k.sqrt();

    // scores[i][j] = how much query i attends to key j (before normalizing).
    let scores = q.matmul(k.transpose()).mul_scalar(scale);

    // softmax over dim 1 (the keys) makes each query's weights sum to 1.
    let weights = softmax(scores.clone(), 1);

    // weighted average of the value rows.
    let out = weights.clone().matmul(v);
    println!("scores  =\n{}", scores.to_data()); // [[0.7071, 0], [0, 0.7071]]
    println!("weights =\n{}", weights.to_data()); // [[0.6698, 0.3302], [0.3302, 0.6698]]
    println!("output  =\n{}", out.to_data()); // [[1.6605, 2.6605], [2.3395, 3.3395]]
}
