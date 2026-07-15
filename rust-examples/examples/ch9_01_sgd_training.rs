use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

type MyBackend = Autodiff<NdArray>;

// A minimal model: a single Linear(1 -> 1) layer, i.e. y = w*x + b.
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

    // Data generated from y = 2x + 1.
    let x = Tensor::<MyBackend, 2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &device);
    let y = Tensor::<MyBackend, 2>::from_floats([[3.0], [5.0], [7.0], [9.0]], &device);

    let mut model: Model<MyBackend> = Model::new(&device);
    let mut optim = SgdConfig::new().init();
    let lr = 0.02;

    // The four-beat training loop: forward -> loss -> backward -> step.
    for _ in 0..3000 {
        let pred = model.forward(x.clone());
        let loss = MseLoss::new().forward(pred, y.clone(), Reduction::Mean);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }

    // Should now approximate y = 2x + 1, so f(5) ~ 11.
    let test = Tensor::<MyBackend, 2>::from_floats([[5.0]], &device);
    println!("prediction for x=5: {}", model.forward(test).to_data()); // ~11
}
