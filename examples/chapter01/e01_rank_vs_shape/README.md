# Rank Is Not Shape

`Tensor<Backend, N>` uses **N** to represent the **rank** (number of axes), **not** the number of elements.

```rust
Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device);
```

A tensor with shape `[5]` is **rank 1**, not rank 5.

> **Rule:** Count the opening brackets. One pair (`[...]`) → rank 1, two pairs (`[[...]]`) → rank 2.
