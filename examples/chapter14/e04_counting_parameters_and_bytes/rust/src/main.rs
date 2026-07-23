use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;

// Chapter 14 — Model footprint, and the no_std picture.
//
// On a microcontroller, memory is THE constraint. Before deploying you want to
// know: how many parameters, and how many bytes at each precision? Module gives
// you num_params(); the rest is arithmetic.
type MyBackend = NdArray;

// A small two-layer MLP so there are real parameters to count.
#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>, // 64 -> 32(weights 64*32 + 32 bias)
    l2: Linear<B>, // 32 -> 10(weights 32*10 + 10 bias)
}

impl<B: Backend> Mlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(64, 32).init(device),
            l2: LinearConfig::new(32, 10).init(device),
        }
    }
}

fn main() {
    let device = Default::default();
    let model: Mlp<MyBackend> = Mlp::new(&device);

    let params = model.num_params();
    println!("parameters   = {params}");
    println!("f32  weights = {} bytes", params * 4);
    println!("f16  weights = {} bytes", params * 2);
    println!("int8 weights = {} bytes (+ tiny per-tensor scale)", params);

    // no_std / embedded notes:
    //  - Burn compiles with no_std for microcontrollers and WASM.
    //  - Weights are baked into the firmware with include_bytes! (ch14_01) and
    //    loaded from RAM with an in-memory recorder — no filesystem.
    //  - The same model code runs on any backend (ch14_02), so you develop on CPU
    //    and deploy to the device unchanged.
    //  - Footprint decides feasibility: int8 quantization (ch14_03) cuts weight
    //    memory ~4x, often the difference between fitting in SRAM/flash or not.
}
