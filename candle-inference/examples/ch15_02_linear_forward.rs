use candle_core::{Device, Tensor};

// Chapter 15 — A linear layer forward pass in Candle.
//
// The inference an "linear + activation" layer does: y = relu(x @ W + b).
// No training here — just the forward pass a served model performs on loaded
// weights. broadcast_add lets the [2] bias add across the [1, 2] result.
fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;

    // One input example with three features: [1, 3].
    let x = Tensor::new(&[[1f32, 2., 3.]], &device)?;

    // "Trained" weights and bias for a 3 -> 2 linear layer.
    let w = Tensor::new(&[[0.1f32, 0.2], [0.3, 0.4], [0.5, 0.6]], &device)?; // [3, 2]
    let b = Tensor::new(&[0.5f32, -0.5], &device)?; // [2]

    let y = x.matmul(&w)?.broadcast_add(&b)?.relu()?;

    println!("output shape = {:?}", y.dims()); // [1, 2]
    println!("output       = {:?}", y.to_vec2::<f32>()?); // [[2.7, 2.3]]
    Ok(())
}
