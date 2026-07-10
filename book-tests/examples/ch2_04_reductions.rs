use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

    let total = x.clone().sum(); // scalar tensor: 10
    let col_sum = x.clone().sum_dim(0); // [[4, 6]]   (keeps the reduced dim)
    let row_mean = x.clone().mean_dim(1); // [[1.5], [3.5]]
    let global_max = x.max(); // 4

    println!("total     = {}", total.to_data()); // [10.0]
    println!("col_sum   = {}", col_sum.to_data()); // [[4.0, 6.0]]
    println!("row_mean  = {}", row_mean.to_data()); // [[1.5], [3.5]]
    println!("global_max= {}", global_max.to_data()); // [4.0]
}
