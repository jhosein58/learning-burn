use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn build() -> (Tensor<Backend, 2>, Tensor<Backend, 2>) {
    let device = Default::default();

    // design matrix X: [n_samples, n_features]
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], &device);
    let y = Tensor::<Backend, 2>::from_floats([[6.0], [0.0], [0.0]], &device);

    let xt = x.clone().transpose();
    let gram = xt.clone().matmul(x); // X^T X
    let xty = xt.matmul(y); // X^T y
    // let w = linalg::solve(gram, xty); // <-- needs upstream burn#1538

    (gram, xty)
}

fn main() {
    let (gram, xty) = build();
    println!("X^T X =\n{}", gram.to_data());
    println!("X^T y =\n{}", xty.to_data());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (gram, xty) = build();

        assert_eq!(gram.dims(), [2, 2]);
        assert_eq!(
            gram.into_data().to_vec::<f32>().unwrap(),
            vec![3.0, 6.0, 6.0, 14.0]
        );

        assert_eq!(xty.dims(), [2, 1]);
        assert_eq!(xty.into_data().to_vec::<f32>().unwrap(), vec![6.0, 6.0]);
    }
}
