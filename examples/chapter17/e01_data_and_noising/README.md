# Example 17.1 — The Data and the Forward Process

2000 points from a ring of 8 Gaussians (radius 2, σ = 0.15), drawn by the chapter's LCG. The linear noise schedule (β from 1e-3 to 0.2 over T = 100 steps, the DDPM schedule rescaled), its √ᾱ_t and √(1−ᾱ_t), and the cosine schedule's ᾱ_t for comparison. The closed-form noising x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε with fixed noise, and the step-by-step chain checked against it.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 1 rust      # from the repo root
cargo run -- -c 17 -e 1 python
cargo run --release -p c17e1  # the runner builds in debug mode; release is much faster
cargo test -p c17e1 --release     # the Rust parity test
pytest examples/chapter17/e01_data_and_noising/python/main.py      # the Python parity test
```

## Parity

x_100 of the three fixed points matches PyTorch to 1e-6, and the chain's standard deviation after 100 steps matches to 1e-5 (the noise comes from the same LCG on both sides).

## Figures

With `CSV_DIR=<dir>` set, the Rust run also writes `c17e1_noising.csv` (1000 points at t = 0, 10, 25, 50, 75, 100) and `c17e1_schedule.csv` for the chapter's figures.
