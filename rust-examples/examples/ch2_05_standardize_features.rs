use burn::backend::NdArray;
use burn::tensor::Tensor;

type B = NdArray;

fn main() {
    let dev = Default::default();

    // x: [n_samples, n_features]
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &dev);

    let mean = x.clone().mean_dim(0); // [1, features]
    println!("mean = {}", mean.to_data());

    let std = x.clone().var(0).sqrt(); // sqrt of the variance
    println!("std  = {}", std.to_data());

    let z = (x - mean) / std; // broadcast
    println!("z    =\n{}", z.to_data());
}
