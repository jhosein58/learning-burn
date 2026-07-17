# Example 1.7 — Getting Your Numbers Back Out

`TensorData` is the bridge both ways. Two methods differ in exactly one respect:
whether the tensor survives.

```rust
let data = t.clone().to_data();     // borrow a view; tensor lives on
let owned = t.into_data();          // consume; tensor is gone
```

- `to_data()` — the tensor lives on (clone it first). Use for a peek.
- `into_data()` — consumes the tensor. Use when you are finished with it.

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c1e7
cargo test --example c1e7
python python.py
pytest python.py
```

## Parity

Both sides read back the values `[1, 2, 3]`.
