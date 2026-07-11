use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::record::CompactRecorder;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

// Chapter 13 — Deploying a model: train, save, reload, infer.
//
// Training and inference are usually separate programs. You train once, write
// the weights to disk, then a lightweight inference process loads them and
// serves predictions. Burn's Recorder handles the serialization; the model
// architecture is just Rust code that both programs share.
//
// Here we do the whole round trip in one file so you can see it end to end.
type Train = Autodiff<NdArray>; // training needs gradients
type Infer = NdArray; // inference does not

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
    let path = std::env::temp_dir().join("burn_ch13_model");
    let recorder = CompactRecorder::new();

    // ---- 1. TRAIN (fit y = 2x + 1) ----
    let x = Tensor::<Train, 2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &device);
    let y = Tensor::<Train, 2>::from_floats([[3.0], [5.0], [7.0], [9.0]], &device);
    let mut model: Model<Train> = Model::new(&device);
    let mut optim = SgdConfig::new().init();
    for _ in 0..3000 {
        let loss = MseLoss::new().forward(model.forward(x.clone()), y.clone(), Reduction::Mean);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(0.02, model, grads);
    }

    // ---- 2. SAVE weights to disk (consumes the trained model) ----
    model
        .save_file(path.clone(), &recorder)
        .expect("failed to save model");
    println!("saved weights to {:?}", path.with_extension("mpk"));

    // ---- 3. LOAD into a fresh inference model (plain backend, no autodiff) ----
    let infer_device = Default::default();
    let loaded: Model<Infer> = Model::new(&infer_device)
        .load_file(path, &recorder, &infer_device)
        .expect("failed to load model");

    // ---- 4. INFER ----
    let test = Tensor::<Infer, 2>::from_floats([[5.0]], &infer_device);
    let pred = loaded.forward(test);
    println!("prediction for x=5 after reload: {}", pred.to_data()); // ~11 (= 2*5 + 1)
}
