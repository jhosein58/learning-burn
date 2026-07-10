use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

// Worked example: z-score each column of a design matrix
// (subtract the column mean, divide by the column standard deviation).
fn main() {
    let device = Default::default();

    // x: [n_samples, n_features]
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);

    let mean = x.clone().mean_dim(0); // [1, features]
    let centered = x.clone() - mean.clone(); // broadcast the mean back over rows
    let var = (centered.clone() * centered.clone()).mean_dim(0); // [1, features]
    let std = var.sqrt();
    let z = centered / std.clone(); // standardized features (clone so we can print std below)

    println!("mean = {}", mean.to_data()); // [[3.0, 4.0]]
    println!("std  = {}", std.to_data()); // ~[[1.63299, 1.63299]]
    println!("z    =\n{}", z.to_data());
    // ~[[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]]
}
