use burn::backend::NdArray;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::{Int, Tensor, TensorData};

type Backend = NdArray;

fn main() {
    let device = Default::default();

    // logits: [batch, num_classes]; cross-entropy takes raw logits (not softmaxed).
    let logits = Tensor::<Backend, 2>::from_floats([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], &device);

    // targets: the correct class index per row.
    let targets = Tensor::<Backend, 1, Int>::from_data(TensorData::from([0i64, 1]), &device);

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let loss = loss_fn.forward(logits, targets);

    println!("cross-entropy = {}", loss.to_data()); // ~0.319
}
