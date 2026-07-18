use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

type Backend = NdArray;

fn main() {
    let device = Default::default();

    // from_floats: most ergonomic for f32 data
    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);

    // from_data: the general path, via a TensorData struct
    let b = Tensor::<Backend, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);

    // a 2-D tensor from nested arrays
    let m = Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    println!("a = {}", a);
    println!("b = {}", b);
    println!("m = {}", m);
}
