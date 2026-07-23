use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

// Chapter 14 — Bare-metal inference: weights embedded in the binary.
//
// A microcontroller has no filesystem. So instead of load_file(), you serialize
// the weights to a byte buffer, bake that buffer into the firmware with
// `include_bytes!("model.bin")`, and load it back from RAM — no disk involved.
// BinBytesRecorder does exactly this: record() -> Vec<u8>, load(bytes) -> weights.

type MyBackend = NdArray;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    linear: Linear<B>,
}
impl<B: Backend> Model<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(4, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}
fn main() {
    let device = Default::default();

    let x = Tensor::<MyBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let model: Model<MyBackend> = Model::new(&device);
    let before = model.forward(x.clone());

    // Serialize weights to an in-memory byte buffer — NO filesystem.
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let bytes: Vec<u8> = recorder
        .record(model.into_record(), ())
        .expect("record to bytes");
    println!("model serialized to {} bytes (held in RAM)", bytes.len());

    // On-device these bytes come from include_bytes!("model.bin"), baked into the
    // firmware. Load them back with zero file I/O:
    let record = recorder
        .load(bytes.clone(), &device)
        .expect("load from bytes");
    let loaded = Model::<MyBackend>::new(&device).load_record(record);

    let after = loaded.forward(x);

    println!("prediction before = {}", before.to_data());
    println!("prediction after  = {}", after.to_data()); // identical
}
