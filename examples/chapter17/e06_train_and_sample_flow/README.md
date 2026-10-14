# Example 17.6 — Train Flow Matching, Then Integrate the ODE

The same network, LCG weights, hand-written Adam and budget as Example 17.4 (6000 steps, batch 256); only the target changes, to the velocity x_1 − x_0. Then 2000 samples by integrating dx/dt = v(x, t) from t = 0 to 1 with Euler and with Heun for 1 to 100 steps, from the same LCG noise as Example 17.4, with the same two scores.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 6 rust      # from the repo root
cargo run -- -c 17 -e 6 python
cargo run --release -p c17e6  # the runner builds in debug mode; release is much faster
cargo test -p c17e6 --release     # the Rust parity test
pytest examples/chapter17/e06_train_and_sample_flow/python/main.py      # the Python parity test
```

## Parity

The first five training losses match torch autograd + `torch.optim.Adam` to 1e-5; after 6000 steps the runs agree to about three decimals in the loss and about 0.01 in the scores.

## Figures

With `CSV_DIR=<dir>` set, the Rust run writes the sample clouds (`c17e6_euler<S>.csv`, `c17e6_heun<S>.csv`).
