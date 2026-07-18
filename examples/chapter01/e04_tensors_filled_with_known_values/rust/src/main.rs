use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let z = Tensor::<Backend, 2>::zeros([2, 3], &device);
    let o = Tensor::<Backend, 2>::ones([2, 3], &device);
    let f = Tensor::<Backend, 2>::full([2, 3], 7.0, &device);

    // identity matrix
    let eye = Tensor::<Backend, 2>::eye(3, &device);

    // a range as an Int tensor: [5, 6, 7, 8, 9]
    let r = Tensor::<Backend, 1, Int>::arange(5..10, &device);

    println!("zeros = {}", z);
    println!("ones = {}", o);
    println!("full = {}", f);
    println!("eye = {}", eye);
    println!("range = {}", r);
}
