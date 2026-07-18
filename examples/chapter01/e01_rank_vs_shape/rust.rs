use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

/// Build the rank-1 tensor holding five values.
///
/// The computation lives in one function so `main` (which prints it) and the
/// parity test (which checks its values) exercise exactly the same code.
fn build() -> Tensor<Backend, 1> {
    let floats = [1.0, 2.0, 3.0, 4.0, 5.0];
    let device = Default::default();

    // CORRECT: a 1-dimensional tensor that happens to hold 5 elements.
    Tensor::<Backend, 1>::from_floats(floats, &device)
}

fn main() {
    let tensor = build();
    println!("tensor = {}", tensor);

    // WRONG: Tensor::<Backend, 5> asks for a 5-DIMENSIONAL tensor, not five
    // values, and will not compile against a flat [f32; 5] array.
    // let tensor = Tensor::<Backend, 5>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity with python.py: same rank, same shape, same values.
    #[test]
    fn matches_pytorch() {
        let tensor = build();

        // rank is 1 (compile-time), shape is [5] (runtime)
        assert_eq!(tensor.dims(), [5]);

        // GOLDEN VALUES — identical to the assertion in python.py
        let values: Vec<f32> = tensor.into_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
