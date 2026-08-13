use burn::backend::NdArray;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::softmax;

// Chapter 12 — Multi-head attention.
//
// One attention is myopic. Multi-head runs several attentions in parallel on
// different learned projections of the input ("heads"), then concatenates them.
// Each head can specialize (one tracks syntax, another long-range links, ...).
//
// Pipeline for input x of shape [seq, d_model]:
//      1. project to Q, K, V with learned linear layers              [seq, d_model]
//      2. split the feature dim into heads: reshape -> swap_dims     [heads, seq, d_k]
//      3. scaled dot-product attention independently per head       (batched matmul)
//      4. concatenate heads back together swap_dims -> reshape      [seq, d_model]
//      5. a final output projection                                 [seq, d_model]

type Backend = NdArray;
fn main() {
    let device = Default::default();

    let seq = 3;
    let d_model = 4;
    let n_heads = 2;
    let d_k = d_model / n_heads; // 2

    // A toy input sequence: 3 tokens, each a 4-dim embedding.
    let x = Tensor::<Backend, 2>::from_floats(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        &device,
    );

    // Learned projections (random init here — the shapes are what matter).
    let wq: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wk: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wv: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wo: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);

    // 1. project.
    let q = wq.forward(x.clone());
    let k = wk.forward(x.clone());
    let v = wv.forward(x.clone());

    // 2. split into heads: [seq, d_model] -> [seq, heads, d_k] -> [heads, seq, d_k]
    let q = q.reshape([seq, n_heads, d_k]).swap_dims(0, 1);
    let k = k.reshape([seq, n_heads, d_k]).swap_dims(0, 1);
    let v = v.reshape([seq, n_heads, d_k]).swap_dims(0, 1);

    // 3. attention per head (batched over the head dimension).
    let scale = 1.0 / (d_k as f64).sqrt();
    let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale); // [heads, seq, seq]
    let weights = softmax(scores, 2); // normalize over keys
    let context = weights.clone().matmul(v); // [heads, seq, d_k]

    // 4. concatenate heads: [heads, seq, d_k] -> [seq, heads, d_k] -> [seq, d_model]
    let concat = context.swap_dims(0, 1).reshape([seq, d_model]);

    // 5. output projection.
    let out = wo.forward(concat);
    println!("inputshape  = {:?}", x.dims()); // [3, 4]
    println!("scoresshape = {:?}", weights.dims()); // [2, 3, 3] (heads, seq, seq)
    println!("outputshape = {:?}", out.dims()); // [3, 4]        (same as input)

    // Sanity: each query's attention weights sum to 1 (softmax), per head.
    println!(
        "weight row sums (should all be 1) =\n{}",
        weights.sum_dim(2).to_data()
    );
}
