use candle_core::{Device, Tensor};

// Chapter 15 — A 2-layer classifier with a softmax head, in Candle.
//
// The full inference path a classifier runs:
//   h      = relu(x @ W1 + b1)     # hidden layer
//   logits = h @ W2                # raw class scores
//   probs  = softmax(logits)       # normalized probabilities that sum to 1
//
// Candle core has no softmax method, so we build it from exp / sum / divide —
// the same definition you saw in the Burn chapters.
fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;

    let x = Tensor::new(&[[1f32, 2., 3.]], &device)?; // [1, 3]

    // Layer 1: 3 -> 2, then ReLU.
    let w1 = Tensor::new(&[[0.1f32, 0.2], [0.3, 0.4], [0.5, 0.6]], &device)?;
    let b1 = Tensor::new(&[0.0f32, 0.0], &device)?;
    let h = x.matmul(&w1)?.broadcast_add(&b1)?.relu()?; // [1, 2]

    // Layer 2: 2 -> 3 (raw logits for 3 classes).
    let w2 = Tensor::new(&[[1.0f32, -1.0, 0.5], [0.5, 1.0, -0.5]], &device)?;
    let logits = h.matmul(&w2)?; // [1, 3]

    // Softmax over the last dimension, by hand.
    let ex = logits.exp()?;
    let denom = ex.sum_keepdim(1)?; // [1, 1]
    let probs = ex.broadcast_div(&denom)?; // [1, 3]

    println!("logits = {:?}", logits.to_vec2::<f32>()?); // ~[[3.6, 0.6, -0.3]]
    println!("probs  = {:?}", probs.to_vec2::<f32>()?); // ~[[0.9346, 0.0465, 0.0189]]
    println!("prob sum = {:?}", probs.sum_keepdim(1)?.to_vec2::<f32>()?); // [[1.0]]
    Ok(())
}
