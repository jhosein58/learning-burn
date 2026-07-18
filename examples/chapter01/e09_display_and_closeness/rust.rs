use burn::backend::NdArray;
use burn::tensor::{Tensor, check_closeness};

type Backend = NdArray;

fn build_full() -> Tensor<Backend, 2> {
    let device = Default::default();
    Tensor::<Backend, 2>::full([2, 3], 0.123456789, &device)
}

/// The largest absolute difference between two nearly-equal vectors.
fn max_abs_diff() -> f32 {
    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device);
    let b = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.001], &device);
    (a - b).abs().max().into_scalar()
}

fn main() {
    let t = build_full();
    println!("{}", t); // full precision
    println!("{:.2}", t); // two decimals via standard Rust formatting

    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device);
    let b = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.001], &device);
    check_closeness(&a, &b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // 0.123456789 stored as f32 rounds to ~0.12345679
        let values: Vec<f32> = build_full().into_data().to_vec().unwrap();
        assert_eq!(values.len(), 6);
        for x in values {
            assert!((x - 0.123_456_79_f32).abs() < 1e-6);
        }

        // the two vectors differ only in the last element, by ~0.001
        let d = max_abs_diff();
        assert!(d > 1e-4 && d < 1e-2, "max abs diff {d}");
    }
}
