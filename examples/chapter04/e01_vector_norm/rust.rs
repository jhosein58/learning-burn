use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type Backend = NdArray;

fn build() -> (Tensor<Backend, 1>, Tensor<Backend, 1>, Tensor<Backend, 1>) {
    let device = Default::default();

    let x = Tensor::<Backend, 1>::from_floats([3.0, 4.0], &device);
    let n2 = linalg::l2_norm(x.clone(), 0); // 5.0
    let unit = linalg::vector_normalize(x, 2.0, 0, 1e-12); // [0.6, 0.8]

    // `det` operates on a BATCH of square matrices, so the input rank must be
    // at least 3: [1, 2, 2] is a batch of one 2x2 matrix. A plain [2, 2] panics.
    //   D1 = D - 1   (rank of the pivot vector used internally)
    //   D2 = D - 2   (rank of the result: the two matrix axes are consumed)
    let m = Tensor::<Backend, 3>::from_floats([[[4.0, 3.0], [6.0, 3.0]]], &device);
    let d = linalg::det::<Backend, 3, 2, 1>(m); // -6.0

    (n2, unit, d)
}

fn main() {
    let (n2, unit, d) = build();
    println!("n2   = {}", n2.to_data());
    println!("unit = {}", unit.to_data());
    println!("det  = {}", d.to_data());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (n2, unit, d) = build();

        assert!((n2.into_scalar() - 5.0).abs() < 1e-6);

        let u: Vec<f32> = unit.into_data().to_vec().unwrap();
        assert!((u[0] - 0.6).abs() < 1e-6 && (u[1] - 0.8).abs() < 1e-6, "unit = {u:?}");

        assert!((d.into_scalar() - (-6.0)).abs() < 1e-5);
    }
}
