use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

// Backpropagation IS the chain rule applied backwards through a computation.
//
// PyTorch counterpart:
//   a = torch.tensor([2.0], requires_grad=True)
//   b = torch.tensor([3.0], requires_grad=True)
//   u = a * b
//   loss = (u * u).sum()
//   loss.backward()
//   a.grad, b.grad   # -> 36, 24
//
// We compose loss = (a * b)^2 and let autodiff walk the chain rule for us.
type Backend = Autodiff<NdArray>;

fn main() {
    let device = Default::default();

    let a = Tensor::<Backend, 1>::from_floats([2.0], &device).require_grad();
    let b = Tensor::<Backend, 1>::from_floats([3.0], &device).require_grad();

    // Forward pass, one node at a time.
    let u = a.clone() * b.clone(); // u = a*b        = 6
    let v = u.clone() * u.clone(); // v = u^2        = 36
    let loss = v.clone().sum(); // loss = v       = 36  (clone so we can print v below)

    // Backward pass: autodiff multiplies the local derivatives along the graph.
    //   dloss/du = 2u = 12
    //   du/da    = b  = 3   ->  dloss/da = 12 * 3 = 36
    //   du/db    = a  = 2   ->  dloss/db = 12 * 2 = 24
    let grads = loss.backward();
    let da = a.grad(&grads).unwrap();
    let db = b.grad(&grads).unwrap();

    println!("u    = {}", u.to_data()); // [6.0]
    println!("v    = {}", v.to_data()); // [36.0]
    println!("loss = {}", loss.to_data()); // [36.0]
    println!("da   = {}", da.to_data()); // [36.0]  == 2u*b
    println!("db   = {}", db.to_data()); // [24.0]  == 2u*a
}
