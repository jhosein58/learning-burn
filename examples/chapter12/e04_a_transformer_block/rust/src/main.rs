use burn::backend::NdArray;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::Tensor;
// Chapter 12 — A transformer encoder block, assembled from scratch.
//
// One block is just four ideas stacked with residual connections:
//
//      x = LayerNorm( x + SelfAttention(x) )     # mix information across tokens
//      x = LayerNorm( x + FeedForward(x) )       # process each token independently
//
// The residual (x + ...) keeps gradients flowing; LayerNorm keeps activations
// well-scaled. Stack N of these and you have a transformer. We write LayerNorm
// by hand so there's no black box.
//
type Backend = NdArray;

// LayerNorm over the last dimension: zero mean, unit variance per token row.
fn layer_norm(x: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
    let mean = x.clone().mean_dim(1); // [seq, 1]
    let centered = x - mean; // broadcast subtract
    let var = (centered.clone() * centered.clone()).mean_dim(1); // [seq, 1]
    centered / var.add_scalar(1e-5).sqrt() // normalize
}
fn main() {
    let device = Default::default();

    let d_model = 4;
    let d_ff = 8;

    let x = Tensor::<Backend, 2>::from_floats(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        &device,
    );
    // Self-attenti
    // on projections.
    let wq: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wk: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wv: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wo: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);

    // Position-wise feed-forward: d_model -> d_ff -> d_model.
    let ff1: Linear<Backend> = LinearConfig::new(d_model, d_ff).init(&device);
    let ff2: Linear<Backend> = LinearConfig::new(d_ff, d_model).init(&device);

    // ---- sub-layer 1: self-attention + residual + norm ----
    let q = wq.forward(x.clone());
    let k = wk.forward(x.clone());
    let v = wv.forward(x.clone());

    let scale = 1.0 / (d_model as f64).sqrt();
    let scores = q.matmul(k.transpose()).mul_scalar(scale); // [seq, seq]
    let weights = softmax(scores, 1);
    let attn = wo.forward(weights.matmul(v)); // [seq, d_model]
    let x = layer_norm(x + attn); // residual, then normalize

    // ---- sub-layer 2: feed-forward + residual + norm ----
    let ff = ff2.forward(relu(ff1.forward(x.clone()))); // [seq, d_model]
    let out = layer_norm(x + ff);
    println!("input shape = {:?}", out.dims()); // [3, 4]
    println!("output shape = {:?}", out.dims()); // [3, 4]   (a block preserves shape)
    println!("output =\n{}", out.to_data());
    // After the final LayerNorm, each row is ~zero-mean:
    println!("row means (≈ 0) =\n{}", out.mean_dim(1).to_data());
}
