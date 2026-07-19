# Element-wise arithmetic

PyTorch vs Burn: addition, multiplication, scalar multiply, and negation—all applied
slot by slot. Burn consumes operands, so reuse requires `.clone()`.

Scalar multiplication uses the named method `mul_scalar` instead of `* 2.0`, because
Rust's operators expect both sides to have the same type.

> `a * b` is element-wise, not matrix multiplication. For the matrix product, use
> `matmul` — exactly the same pitfall as NumPy's `*` vs `@`.
