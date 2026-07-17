use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;
type F2 = Tensor<Backend, 2>;
type I1 = Tensor<Backend, 1, Int>;

fn build() -> (F2, F2, F2, F2, I1) {
    let device = Default::default();

    // Annotate the binding; let it carry the type (no turbofish on the call).
    let z: F2 = Tensor::zeros([2, 3], &device);
    let o: F2 = Tensor::ones([2, 3], &device);
    let f: F2 = Tensor::full([2, 3], 7.0, &device);
    let eye: F2 = Tensor::eye(3, &device);
    let r: I1 = Tensor::arange(5..10, &device); // [5, 6, 7, 8, 9]

    (z, o, f, eye, r)
}

fn main() {
    let (z, o, f, eye, r) = build();
    println!("zeros = {}", z);
    println!("ones = {}", o);
    println!("full = {}", f);
    println!("eye = {}", eye);
    println!("range = {}", r);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (z, o, f, eye, r) = build();

        assert_eq!(z.into_data().to_vec::<f32>().unwrap(), vec![0.0; 6]);
        assert_eq!(o.into_data().to_vec::<f32>().unwrap(), vec![1.0; 6]);
        assert_eq!(f.into_data().to_vec::<f32>().unwrap(), vec![7.0; 6]);
        assert_eq!(
            eye.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(r.into_data().to_vec::<i64>().unwrap(), vec![5, 6, 7, 8, 9]);
    }
}
