# Example 15.6 — Greedy, Temperature, Top-k and Top-p by Hand

Four decoding strategies as functions from a logits vector to a probability vector, on the five logits [2, 1, 0.5, 0, -1]: greedy, temperature (T = 1, 0.5, 2), top-k (k = 2) and top-p (p = 0.8). Tokens are drawn by the inverse CDF at uniforms from the LCG, so the draws are deterministic.

```rust
let before = sorted.clone().cumsum(0) - sorted.clone(); // mass ahead of each token
let kept: i32 = before.lower_elem(p).int().sum().into_scalar();
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 15 -e 6 rust      # from the repo root
cargo run -- -c 15 -e 6 python
cargo test -p c15e6 --release     # the Rust parity test
pytest examples/chapter15/e06_sampling/python/main.py      # the Python parity test
```

## Parity

The probabilities match torch to 1e-6, and the counts of 10,000 draws per strategy are identical.
