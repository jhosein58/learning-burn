# Example 1.2 — Creating Tensors from Data

Two standard ways to create a tensor in Burn, and their PyTorch equivalents.

## What this example shows

```rust
Tensor::from_floats(...)   // ergonomic for floating-point arrays
Tensor::from_data(...)     // generic constructor via TensorData
```

Both create the same tensor; they differ only in how the input is provided.
Every tensor is created on an explicit **device**.

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version. |
| `python.py` | PyTorch equivalent. |

## Run & test

```bash
cargo run  --example c1e2
python python.py
```

## Parity

Both sides assert the same values: `a = [1, 2, 3]` and
`m = [[1, 2, 3], [4, 5, 6]]` with shape `[2, 3]`.
