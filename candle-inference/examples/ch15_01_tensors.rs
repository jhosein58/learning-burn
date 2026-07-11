use candle_core::{Device, Tensor};

// Chapter 15 — Inference with Candle: tensor basics.
//
// Candle is HuggingFace's minimalist Rust ML framework, widely used to run
// pretrained models (Llama, Whisper, ...). Its tensor API feels close to Burn,
// with two visible differences:
//   * operations return Result (shapes are checked at runtime) -> you use `?`
//   * you choose a Device explicitly (Device::Cpu; new_cuda/new_metal for GPU)
fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;

    let a = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?; // [2, 3]
    let b = Tensor::new(&[[1f32, 1., 1.], [2., 2., 2.]], &device)?;

    // Elementwise ops on &Tensor return Result, hence the `?`.
    let sum = (&a + &b)?;
    let prod = (&a * &b)?;

    // Matmul: [2, 3] @ [3, 2] -> [2, 2].
    let c = Tensor::new(&[[1f32, 0.], [0., 1.], [1., 1.]], &device)?;
    let mm = a.matmul(&c)?;

    println!("a shape = {:?}", a.dims()); // [2, 3]
    println!("a + b   = {:?}", sum.to_vec2::<f32>()?); // [[2,3,4],[6,7,8]]
    println!("a * b   = {:?}", prod.to_vec2::<f32>()?); // [[1,2,3],[8,10,12]]
    println!("a @ c   = {:?}", mm.to_vec2::<f32>()?); // [[4,5],[10,11]]
    Ok(())
}
