use burn::backend::NdArray;
use burn::tensor::Tensor;

type B = NdArray;

fn main() {
    let dev = Default::default();

    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev);
    let total = x.clone().sum(); // scalar tensor 10
    let col_sum = x.clone().sum_dim(0); // [[4, 6]]
    let row_mean = x.clone().mean_dim(1); // [[1.5],[3.5]]

    let (m, idx) = x.max_dim_with_indices(1);

    println!("total     = {}", total.to_data()); // [10.0]
    println!("col_sum   = {}", col_sum.to_data()); // [[4.0, 6.0]]
    println!("row_mean  = {}", row_mean.to_data()); // [[1.5], [3.5]]
    println!("m-idx     = {} - {}", m.to_data(), idx.to_data());
}
