use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn build() -> Tensor<Backend, 1> {
    let device = Default::default();
    let input = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);

    // Min-max normalisation needs the input three times, so `.clone()` says so.
    let min = input.clone().min();
    let max = input.clone().max();
    (input.clone() - min.clone()).div(max - min)
}

fn main() {
    println!("{}", build().to_data());
    // [0.0, 0.33333334, 0.6666667, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let values: Vec<f32> = build().into_data().to_vec().unwrap();
        let expected = [0.0f32, 1.0 / 3.0, 2.0 / 3.0, 1.0];

        assert_eq!(values.len(), 4);
        for i in 0..4 {
            assert!(
                (values[i] - expected[i]).abs() < 1e-6,
                "index {i}: got {}, want {}",
                values[i],
                expected[i]
            );
        }
    }
}
