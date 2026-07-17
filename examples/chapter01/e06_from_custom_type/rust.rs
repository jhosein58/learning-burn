use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

type Backend = NdArray;

struct BodyMetrics {
    age: i8,
    height: i16,
    weight: f32,
}

fn build(bmi: &BodyMetrics) -> Tensor<Backend, 1> {
    let device = Default::default();
    // Take the fields you care about, convert to a common numeric type with
    // `as f32`, and let TensorData carry them across.
    let data = TensorData::from([bmi.age as f32, bmi.height as f32, bmi.weight]);
    Tensor::<Backend, 1>::from_data(data, &device)
}

fn main() {
    let bmi = BodyMetrics {
        age: 25,
        height: 180,
        weight: 80.0,
    };
    println!("t = {}", build(&bmi));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let bmi = BodyMetrics {
            age: 25,
            height: 180,
            weight: 80.0,
        };
        let values: Vec<f32> = build(&bmi).into_data().to_vec().unwrap();
        assert_eq!(values, vec![25.0, 180.0, 80.0]);
    }
}
