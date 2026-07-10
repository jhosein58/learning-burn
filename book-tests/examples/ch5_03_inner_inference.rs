use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

// There is no `no_grad` block in Burn. For inference you either don't use the
// `Autodiff` wrapper at all, or call `inner()` to drop to the base backend and
// skip graph tracking.
fn main() {
    let device = Default::default();

    let x = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);

    // Peel off the autodiff graph -> a plain NdArray tensor, no tracking.
    let plain = x.inner();
    let y = plain.add_scalar(5.0); // cheap inference op

    println!("{}", y.to_data()); // [6.0, 7.0, 8.0]
}
