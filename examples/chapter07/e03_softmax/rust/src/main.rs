use burn::backend::NdArray;
use burn::tensor::{Tensor, activation};

type B = NdArray;

fn main() {
    let dev = Default::default();

    // logits: [batch, num_classes]
    let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &dev);

    // softmax over the class dimension; each row sums to 1.
    let probs = activation::softmax(logits.clone(), 1);

    println!("softmax = {}", probs.to_data()); // [0.09003, 0.24473, 0.66524]
    println!("sum     = {}", probs.sum().to_data()); // [1.0]
}
