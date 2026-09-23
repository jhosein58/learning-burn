# Example 15.6 — What the Filters Learned, and an int8 CNN

Trains exactly as 15.5, prints the four 3x3 filters of the first layer, then quantises every weight tensor to int8 with Chapter 14's symmetric scheme.

```rust
fn int8_round_trip<B: Backend, const D: usize>(w: Tensor<B, D>) -> (Tensor<B, D>, f32) {
    let scale = w.clone().abs().max().into_scalar().elem::<f32>() / 127.0;
    let codes = w.div_scalar(scale).round().clamp(-127.0, 127.0);
    (codes.mul_scalar(scale), scale)
}
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
pytest examples/chapter15/e06_filters_int8/python/main.py      # the Python parity test
```

## Parity

The learned filters match PyTorch's to 2e-3. Both models score 80/80 before and after quantisation, and the size drops from 1,488 bytes to 432.
