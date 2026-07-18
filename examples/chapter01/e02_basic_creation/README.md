# Creating Tensors

Burn provides two ways to create tensors from existing data:

- `Tensor::from_floats()` — simple and ergonomic.
- `Tensor::from_data()` — the general constructor using `TensorData`.

Every tensor is created on a specific **device**, which is passed explicitly.

> `from_floats()` stores values as **f32**, even though Rust float literals are `f64` by default.
