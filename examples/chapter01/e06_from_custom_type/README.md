# Example 1.6 — A Tensor From Your Own Struct

Real data arrives as your own types, not array literals. Convert the fields to a
common numeric type and let `TensorData` carry them into a tensor.

```rust
let data = TensorData::from([bmi.age as f32, bmi.height as f32, bmi.weight]);
Tensor::<Backend, 1>::from_data(data, &device)
```

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent (a `@dataclass`) + `pytest` test. |

## Run & test

```bash
cargo run  --example c1e6
cargo test --example c1e6
python python.py
pytest python.py
```

## Parity

Both sides assert `[25, 180, 80]` — age, height, weight as `f32`.
