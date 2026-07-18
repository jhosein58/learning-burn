use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};

type Backend = NdArray;

fn main() {
    let device = Default::default();
    let arr: [i32; 6] = [1, 2, 3, 4, 5, 6];

    // an Int tensor built from a slice of the array
    let ints = Tensor::<Backend, 1, Int>::from_data(TensorData::from(&arr[0..3]), &device);

    println!("ints = {}", ints);
}
