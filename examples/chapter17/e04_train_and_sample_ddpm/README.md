# Example 17.4 — Train a DDPM, Then Sample It Two Ways

6000 training steps (batch 256) of the noise-prediction MLP, with the backward pass of Example 17.2 and a hand-written Adam (lr 2e-3, decayed linearly to 0; β = 0.9, 0.999; ε = 1e-8). No autodiff and no optimizer. Then 2000 samples by DDPM ancestral sampling (100 network calls) and by DDIM with 5, 10, 20, 50 and 100 steps, all from the same LCG noise, scored by a 2-D histogram distance (tv) and the mean nearest-neighbour distance (nn) to 2000 fresh data points.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 4 rust      # from the repo root
cargo run -- -c 17 -e 4 python
cargo run --release -p c17e4  # the runner builds in debug mode; release is much faster
cargo test -p c17e4 --release     # the Rust parity test
pytest examples/chapter17/e04_train_and_sample_ddpm/python/main.py      # the Python parity test
```

## Parity

The first five training losses match torch autograd + `torch.optim.Adam` to 1e-5. After 6000 steps the two runs still agree to about three decimals in the loss and about 0.01 in the sample scores (f32 rounding differences grow slowly).

## Figures

With `CSV_DIR=<dir>` set, the Rust run writes the sample clouds (`c17e4_ddpm100.csv`, `c17e4_ddim<S>.csv`).
