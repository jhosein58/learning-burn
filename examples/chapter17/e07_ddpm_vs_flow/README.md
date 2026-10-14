# Example 17.7 — DDPM and Flow Matching, Head to Head

Both models trained with the same data generator, network, weights, Adam and number of steps. It times the training, scores the samples at checkpoints with a fixed budget of 20 network calls (DDIM 20 steps, Euler 20, Heun 10), measures how far a coarse solve lands from a 100-step solve of the same ODE, and compares the path lengths of the two samplers with those of the straight training pairs.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 7 rust      # from the repo root
cargo run -- -c 17 -e 7 python
cargo run --release -p c17e7  # the runner builds in debug mode; release is much faster
cargo test -p c17e7 --release     # the Rust parity test
pytest examples/chapter17/e07_ddpm_vs_flow/python/main.py      # the Python parity test
```

## Parity

The first three training losses of each model match PyTorch to 1e-5. The later numbers agree to about 0.01 (scores) and a few percent (solver errors); timings depend on the machine.

## Figures

With `CSV_DIR=<dir>` set, the Rust run writes 64 paths of each sampler (`c17e7_paths_ddim.csv`, `c17e7_paths_flow.csv`) and 64 training pairs (`c17e7_pairs.csv`).
