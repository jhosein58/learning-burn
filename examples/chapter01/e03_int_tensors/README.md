# Example 1.3 — Integer Tensors

When you need whole numbers — class labels, indices, counts — fill the third
type slot with `Int`.

```rust
Tensor::<Backend, 1, Int>::from_ints([1, 2, 3], &device)
```

This mirrors the float example (e02) exactly: only the `Int` slot and
`from_ints` (instead of `from_floats`) change.

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c1e3
cargo test --example c1e3
python python.py
pytest python.py
```

## Parity

Both sides assert values `[1, 2, 3]` at 64-bit integer precision
(Burn NdArray `Int` = `i64`; torch default = `int64`).
