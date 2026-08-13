use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

// Chapter 14 — Backend-agnostic inference.
//
// Burn's core idea: your model and inference code are generic over the Backend
// trait `B`. The exact same code compiles for CPU (NdArray), GPU (Wgpu/CUDA), or
// an embedded backend — you only pick the concrete type at the call site. That's
// what lets you develop on a laptop and deploy to a very different device.

#[derive(Module, Debug)]
struct Classifier<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> Classifier<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(3, 2).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

// Inference written ONCE, generic over B. Nothing here mentions a concrete device.
fn run_inference<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
    let model = Classifier::<B>::new(device);
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], device);
    model.forward(x)
}

fn main() {
    // CPU today:
    let out = run_inference::<NdArray>(&Default::default());

    println!("output shape = {:?}", out.dims()); // [1, 2]
    println!("output = {}", out.to_data());

    // To run the SAME function on a GPU:
    //      run_inference::<burn::backend::Wgpu>(&Default::default());
    // For embedded, substitute a no_std backend. The function body never changes.
}
