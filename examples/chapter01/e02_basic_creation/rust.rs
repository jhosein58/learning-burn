use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

type Backend = NdArray;

fn build_a() -> Tensor<Backend, 1> {
    let device = Default::default();
    // from_floats: most ergonomic for f32 data.
    Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device)
}

fn build_m() -> Tensor<Backend, 2> {
    let device = Default::default();
    Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)
}

fn main() {
    let device = Default::default();

    let a = build_a();
    // from_data: the general path, via a TensorData struct (same values as `a`).
    let b = Tensor::<Backend, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);
    let m = build_m();

    println!("a = {}", a);
    println!("b = {}", b);
    println!("m = {}", m);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let a: Vec<f32> = build_a().into_data().to_vec().unwrap();
        assert_eq!(a, vec![1.0, 2.0, 3.0]);

        let m = build_m();
        assert_eq!(m.dims(), [2, 3]);
        let m_flat: Vec<f32> = m.into_data().to_vec().unwrap();
        assert_eq!(m_flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
