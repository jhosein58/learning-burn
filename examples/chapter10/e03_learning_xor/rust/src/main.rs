use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

// The classic backprop demonstration: XOR is NOT linearly separable, so a single
// Linear layer can never solve it. A 2-layer net (Linear -> tanh -> Linear) can,
// but only because backprop pushes gradients through the hidden layer.
//
// PyTorch counterpart: nn.Sequential(nn.Linear(2,8), nn.Tanh(), nn.Linear(8,1))
// trained with MSELoss + SGD.
type MyBackend = Autodiff<NdArray>;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>, // 2 -> 8 hidden units
    l2: Linear<B>, // 8 -> 1 output
}

impl<B: Backend> Mlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(2, 8).init(device),
            l2: LinearConfig::new(8, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.l1.forward(x).tanh(); // hidden layer + nonlinearity
        self.l2.forward(h)
    }
}

fn main() {
    let device = Default::default();

    // The four XOR examples.
    let x = Tensor::<MyBackend, 2>::from_floats(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        &device,
    );

    let y = Tensor::<MyBackend, 2>::from_floats([[0.0], [1.0], [1.0], [0.0]], &device);

    let mut model: Mlp<MyBackend> = Mlp::new(&device);
    let mut optim = SgdConfig::new().init();

    // Keep the step small: tanh saturates, and a large lr makes full-batch SGD
    // oscillate and blow up to NaN. 0.1 is stable here.
    let lr = 0.1;
    for epoch in 0..40000 {
        let pred = model.forward(x.clone());
        let loss = MseLoss::new().forward(pred, y.clone(), Reduction::Mean);

        if epoch % 8000 == 0 {
            println!("epoch {:>5}  loss = {}", epoch, loss.clone().to_data());
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }

    // After training, predictions should snap toward the XOR truth table: 0,1,1,0.
    let preds = model.forward(x.clone());

    println!(
        "\nfinal predictions (target 0, 1, 1, 0):\n{}",
        preds.to_data()
    );
    // Note: random init means a rare run can settle in a bad local minimum.
    // If the four outputs are not close to 0,1,1,0, just run it again.
}
