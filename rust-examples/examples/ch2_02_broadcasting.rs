use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    // Add a per-column bias to every row of a [2, 3] matrix.
    let m = Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let bias = Tensor::<Backend, 1>::from_floats([100.0, 200.0, 300.0], &device);

    // `unsqueeze` raises the length-3 vector to [1, 3] so its rank matches the
    // matrix; it then broadcasts over the 2 rows.
    let out = m + bias.unsqueeze();

    println!("{}", out.to_data());
    // [[101.0, 202.0, 303.0], [104.0, 205.0, 306.0]]
}
