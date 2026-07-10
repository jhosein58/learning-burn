use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let m = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let eye = Tensor::<Backend, 2>::eye(2, &device);

    // trace = sum of the diagonal = sum(m elementwise-* identity).
    let trace = (m.clone() * eye.clone()).sum(); // 1 + 4 = 5

    // Batched matmul: 4 matrices [2,3] times 4 matrices [3,2] -> [4, 2, 2].
    // Each output element is the sum of 3 ones = 3.
    let ba = Tensor::<Backend, 3>::ones([4, 2, 3], &device);
    let bb = Tensor::<Backend, 3>::ones([4, 3, 2], &device);
    let bc = ba.matmul(bb); // [4, 2, 2]

    println!("identity =\n{}", eye.to_data());
    println!("trace    = {}", trace.to_data()); // [5.0]
    println!("batched dims = {:?}", bc.dims()); // [4, 2, 2]
    println!("batched  =\n{}", bc.to_data());
}
