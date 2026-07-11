use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

    let total = x.clone().sum(); // scalar tensor, shape [1]
    let col_sum = x.clone().sum_dim(0); // shape [1, 2] — the reduced dim is kept
    let row_mean = x.clone().mean_dim(1); // shape [2, 1] — the reduced dim is kept
    let global_max = x.max(); // scalar tensor, shape [1]

    // NOTE: to_data() prints the values FLAT. The shape is a separate fact —
    // print the tensor itself (not .to_data()) if you want to see it.
    println!("total     = {}", total.to_data()); // [10.0]
    println!("col_sum   = {}", col_sum.to_data()); // [4.0, 6.0]   (shape [1, 2])
    println!("row_mean  = {}", row_mean.to_data()); // [1.5, 3.5]   (shape [2, 1])
    println!("global_max= {}", global_max.to_data()); // [4.0]
}
