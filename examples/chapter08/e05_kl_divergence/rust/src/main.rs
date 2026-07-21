use burn::backend::NdArray;
use burn::nn::loss::{KLDivLossConfig, Reduction};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{Tensor, TensorData};

type B = NdArray;

fn main() {
    let device = Default::default();

    let student_logits = Tensor::<B, 2>::from_data(
        TensorData::from([
            [2.0, 0.5, -1.0, 0.0],
            [0.1, 1.2, 0.3, -0.5],
            [-0.2, 0.4, 1.5, 0.8],
        ]),
        &device,
    );

    let teacher_logits = Tensor::<B, 2>::from_data(
        TensorData::from([
            [2.5, 0.2, -0.8, 0.1],
            [-0.1, 1.6, 0.2, -0.3],
            [0.0, 0.7, 1.2, 0.4],
        ]),
        &device,
    );

    // KL divergence expects log-probabilities for predictions.
    let student = log_softmax(student_logits, 1); // KLDiv wants LOG-probabilities
    let teacher = softmax(teacher_logits, 1); // the target is a distribution

    let loss = KLDivLossConfig::new()
        .init()
        .forward(student, teacher, Reduction::Auto); // Auto == batchmean

    println!("loss = {}", loss.into_scalar()); // ~0.034
}
