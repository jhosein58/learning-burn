# Example 1.8 — Ownership, and why you type `.clone()`

Tensor operations **consume** their inputs. Min-max normalisation needs the
input three times, so the code says so — three `.clone()`s.

```rust
let min = input.clone().min();
let max = input.clone().max();
let input = (input.clone() - min.clone()).div(max - min);
```

A `.clone()` on a Burn tensor bumps a reference count, it does not deep-copy the
data — the same bookkeeping CPython does silently.

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c1e8
cargo test --example c1e8
python python.py
pytest python.py
```

## Parity

Both sides normalise `[1, 2, 3, 4]` to `[0, 1/3, 2/3, 1]`, compared with a
`1e-6` tolerance (never `==` on floats).
