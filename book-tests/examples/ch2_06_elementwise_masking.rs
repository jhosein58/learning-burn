use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    let x = Tensor::<Backend, 1>::from_floats([-1.0, 0.0, 2.0], &device);

    // Chained transcendental ops, and clamping.
    let softplus = x.clone().exp().log1p(); // log(1 + e^x)
    let clamped = x.clone().clamp(0.0, 1.0); // torch.clamp

    // A comparison produces a Bool tensor that drives masking.
    let mask = x.clone().greater_elem(0.0); // [false, false, true]
    let replacement = Tensor::<Backend, 1>::from_floats([9.0, 9.0, 9.0], &device);
    let picked = x.mask_where(mask, replacement); // where mask -> 9, else x

    println!("softplus = {}", softplus.to_data()); // [0.31326, 0.69315, 2.12693]
    println!("clamped  = {}", clamped.to_data()); // [0.0, 0.0, 1.0]
    println!("picked   = {}", picked.to_data()); // [-1.0, 0.0, 9.0]
}
