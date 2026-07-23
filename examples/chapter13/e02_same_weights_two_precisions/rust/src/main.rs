use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::record::{CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
// Chapter 13 — Recorder formats: precision vs file size.
//
// A "recorder" decides how weights are serialized. Two msgpack recorders differ
// only in numeric precision:
//      * NamedMpkFileRecorder<FullPrecisionSettings> -> f32: exact weights
//      * CompactRecorder (= NamedMpk + HalfPrecisionSettings) -> f16: weights are
//        rounded to half precision, so predictions shift slightly
//
// Saving the SAME model both ways shows the precision trade-off directly: full
// precision reloads bit-for-bit; compact loses a little accuracy. f16 also
// halves the WEIGHT bytes — but that only shrinks the file on a real model with
// thousands of parameters. On this 2-parameter toy the file is almost all format
// overhead, so the compact file is not smaller here (see the printout).
//
type MyBackend = NdArray;

// NOTE: the Module derive requires the backend generic to be named `B`.
#[derive(Module, Debug)]
struct Model<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(1, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

fn main() {
    let device = Default::default();

    let x = Tensor::<MyBackend, 2>::from_floats([[2.0]], &device);
    let model: Model<MyBackend> = Model::new(&device);
    let original = model.forward(x.clone());

    // --- Full precision (f32): exact round-trip ---
    let full = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let full_path = std::env::temp_dir().join("ch13_full");
    model
        .clone()
        .save_file(full_path.clone(), &full)
        .expect("save full");
    let from_full: Model<MyBackend> = Model::new(&device)
        .load_file(full_path.clone(), &full, &device)
        .expect("load full");
    let pred_full = from_full.forward(x.clone());

    // --- Compact (f16): smaller file, slight precision loss ---
    let compact = CompactRecorder::new();
    let compact_path = std::env::temp_dir().join("ch13_compact");
    model
        .save_file(compact_path.clone(), &compact)
        .expect("save compact");
    let from_compact: Model<MyBackend> = Model::new(&device)
        .load_file(compact_path.clone(), &compact, &device)
        .expect("load compact");
    let pred_compact = from_compact.forward(x);
    let full_size = std::fs::metadata(full_path.with_extension("mpk"))
        .map(|m| m.len())
        .unwrap_or(0);
    let compact_size = std::fs::metadata(compact_path.with_extension("mpk"))
        .map(|m| m.len())
        .unwrap_or(0);
    println!("original prediction   = {}", original.to_data());
    println!("full-precision reload = {}", pred_full.to_data()); // exactly equal
    println!("compact (f16)  reload = {}", pred_compact.to_data()); // very close
    println!("full    (f32)  file   = {full_size} bytes");
    println!("compact (f16)  file   = {compact_size} bytes");
    // On this tiny model format overhead dominates, so f16 does not shrink the
    // file. The size win appears at scale, where weight bytes dominate.
}
