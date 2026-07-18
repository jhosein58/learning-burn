# Example 1.5 — Random Tensors

Burn takes one `random` function and asks which distribution you want, where
PyTorch has `rand` and `randn`.

```rust
Tensor::random([2, 3], Distribution::Default, &device)          // uniform [0,1)
Tensor::random([2, 3], Distribution::Normal(0.0, 1.0), &device) // standard normal
```

## Parity is statistical, not exact

Burn and PyTorch use **different RNG algorithms**, so their draws never match
element-for-element — even with the same seed. Instead, each side checks its own
draw against the **same distributional contract**, over a large sample
(`N = 100_000`) with a loose tolerance so the test never flakes:

- **uniform**: every value in `[0, 1)`; mean ≈ 0.5
- **normal**: mean ≈ 0; std ≈ 1

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + statistical `#[test]`s. |
| `python.py` | PyTorch equivalent + matching `pytest` tests. |

## Run & test

```bash
cargo run  --example c1e5
cargo test --example c1e5
python python.py
pytest python.py
```
