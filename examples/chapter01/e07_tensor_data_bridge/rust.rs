use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn build() -> Tensor<Backend, 1> {
    let device = Default::default();
    Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device)
}

fn main() {
    let t = build();

    // inspect without consuming (clone first, then borrow a view)
    let data = t.clone().to_data();
    let values: &[f32] = data.as_slice().unwrap();
    println!("values (borrowed view) = {:?}", values);

    // or consume it when you are done
    let owned = t.into_data();
    println!("owned data = {:?}", owned);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let values: Vec<f32> = build().into_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
}
