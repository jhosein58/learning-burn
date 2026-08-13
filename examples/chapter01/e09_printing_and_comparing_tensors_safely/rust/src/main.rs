use burn::backend::NdArray;
use burn::tensor::{Tensor, check_closeness};

type Backend = NdArray;

fn main() {
    let dev = Default::default();

    let tensor = Tensor::<Backend, 2>::full([2, 3], 0.123456789, &dev);
    println!("{}", tensor);

    // control precision with standard Rust formatting
    println!("{:.2}", tensor);

    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &dev);
    let b = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.001], &dev);

    check_closeness(&a, &b);
}
