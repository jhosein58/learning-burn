use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

// Worked example: a linear layer forward pass, y = X * W^T + b.
// This is exactly what `nn.Linear` computes under the hood.
fn main() {
    let device = Default::default();

    // X: [batch, in], W: [out, in], b: [out]
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device); // [2, 2]
    let w = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], &device); // [3, 2]
    let b = Tensor::<Backend, 1>::from_floats([10.0, 20.0, 30.0], &device); // [3]

    // matmul with the transposed weights, then broadcast-add the bias over the batch.
    let y = x.matmul(w.transpose()) + b.unsqueeze(); // [2, 3]

    println!("{}", y.to_data());
    // [[11.0, 22.0, 33.0], [13.0, 24.0, 37.0]]
}
