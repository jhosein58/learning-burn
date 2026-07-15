use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type B = NdArray;

fn main() {
    let dev = Default::default();

    let m = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev);
    let eye = Tensor::<B, 2>::eye(3, &dev); // identity

    let lo = m.clone().tril(0); // lower-triangular part
    let up = m.clone().triu(0); // upper-triangular part
    let tr: Tensor<B, 1> = linalg::trace(m); // sum of the diagonal

    // Batched matmul: 8 matrices [2,3] times 8 matrices [3,4] -> [8, 2, 4]
    let ba = Tensor::<B, 3>::zeros([8, 2, 3], &dev);
    let bb = Tensor::<B, 3>::zeros([8, 3, 4], &dev);
    let bc = ba.matmul(bb); // [8, 2, 4]

    println!("identity =\n{}", eye.to_data());
    println!("lo       = {}", lo.to_data());
    println!("up       = {}", up.to_data());
    println!("trace    = {}", tr.to_data());
    println!("batched dims = {:?}", bc.dims());
    println!("batched  =\n{}", bc.to_data());
}
