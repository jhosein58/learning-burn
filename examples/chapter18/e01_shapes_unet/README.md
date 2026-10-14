# Example 18.1 — A conditional U-Net for flow matching

Chapter 13's 12x12 shapes scaled to [-1, 1], a sinusoidal time embedding, and a small U-Net (`#[derive(Module)]`: GroupNorm residual blocks, one stride-2 down path, a transposed-convolution up path, a skip connection, and time + class conditioning). Built with fixed weights, it computes the velocity for four training items, the flow-matching loss, and one classifier-free-guided Euler step from fixed noise. The parameter count is checked against a formula.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 18 -e 1 rust      # from the repo root
cargo run -- -c 18 -e 1 python
cargo test -p c18e1 --release     # the Rust test
pytest examples/chapter18/e01_shapes_unet/python/main.py
```

## Parity

The parameter count (48,689) is identical; velocities, loss and the guided step match PyTorch to 1e-4. `python main.py --expected` prints the numbers the Rust test checks.
