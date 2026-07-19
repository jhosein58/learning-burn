# Vector Multiplication: matvec, dot, outer

PyTorch vs Burn: three ways to multiply with a vector. `linalg::matvec` pushes a
vector through a matrix, `dot` collapses two vectors into a scalar, and `outer` grows
an axis to produce a matrix.

These live in `burn::tensor::linalg` — they are imported functions, not methods.

> Operations that only make sense at specific ranks live in `linalg`. It keeps the
> `Tensor` method list honest and makes the shape semantics explicit.
