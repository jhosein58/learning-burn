use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};

type Backend = NdArray;

fn main() {
    let device = Default::default();
    let arr: [i32; 6] = [1, 2, 3, 4, 5, 6];

    // an Int tensor built from a slice of the array with from_ints
    let from_ints = Tensor::<Backend, 1, Int>::from_ints(&arr[0..3], &device);

    // an Int tensor built from a slice of the array with TensorData
    let tensor_data_ints = Tensor::<Backend, 1, Int>::from_data(TensorData::from(&arr[0..3]), &device);

    // a 2-D tensor from nested arrays
    let two_d_ints = Tensor::<Backend, 2, Int>::from_ints([[1, 2, 3], [-1, 5, -6]], &device);

    println!("from_ints = {}", from_ints);
    println!("tensor_data_ints = {}", tensor_data_ints);
    println!("two_d_ints = {}", two_d_ints);
}
