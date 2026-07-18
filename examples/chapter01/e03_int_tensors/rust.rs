use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;

fn build() -> Tensor<Backend, 1, Int> {
    let device = Default::default();
    // Exactly like the float tensor in e02, with one change: the third type
    // slot is Int, and the builder is from_ints instead of from_floats.
    Tensor::<Backend, 1, Int>::from_ints([1, 2, 3], &device)
}

fn main() {
    println!("ints = {}", build());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let t = build();
        assert_eq!(t.dims(), [3]);

        // NdArray's default integer element is i64.
        let values: Vec<i64> = t.into_data().to_vec().unwrap();
        assert_eq!(values, vec![1, 2, 3]);
    }
}
