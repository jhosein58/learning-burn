# Example 17.2 — The Noise-Prediction Loss and Its Gradient

The noise-prediction network ε_θ(x_t, t): an MLP 18 → 128 → 128 → 2 with relu, fed x_t and a 16-feature sinusoidal embedding of t/T, with fixed weights drawn by the LCG (PyTorch's nn.Linear initialisation). The loss L = (1/N) Σ‖ε̂ − ε‖² on the first training batch of Example 17.4 (N = 256), its gradient dL/dε̂ = (2/N)(ε̂ − ε), and the backward pass through all three layers, written by hand with tensor ops.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 2 rust      # from the repo root
cargo run -- -c 17 -e 2 python
cargo run --release -p c17e2  # the runner builds in debug mode; release is much faster
cargo test -p c17e2 --release     # the Rust parity test
pytest examples/chapter17/e02_ddpm_loss_and_gradient/python/main.py      # the Python parity test
```

## Parity

The hand-written gradients equal Burn's autodiff exactly (max difference 0.0 over all 19 202 gradients), and the loss and selected gradients match torch `.backward()` to 1e-6.
