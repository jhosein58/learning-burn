use burn::backend::{Autodiff, NdArray};
use burn::nn::{Linear, LinearConfig};
use burn::optim::momentum::MomentumConfig;
use burn::optim::{AdamConfig, AdamWConfig, SgdConfig};

type B = Autodiff<NdArray>;

fn main() {
    let device = Default::default();

    // A small trainable model used only to specify the optimizer module type.
    let _model = LinearConfig::new(1, 1).init::<B>(&device);

    // Plain stochastic gradient descent.
    let _sgd = SgdConfig::new().init::<B, Linear<B>>();

    // SGD with momentum.
    let _sgd_m = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new()))
        .init::<B, Linear<B>>();

    // Adam uses adaptive learning rates.
    let _adam = AdamConfig::new().init::<B, Linear<B>>();

    // AdamW uses decoupled weight decay.
    let _adamw = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, Linear<B>>();

    println!("Optimizers created: SGD, SGD with momentum, Adam, AdamW");
}
