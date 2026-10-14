# Example 17.3 — One DDPM Step and One DDIM Step

One DDPM ancestral step (t = 50 → 49) and one deterministic DDIM step (t = 50 → 40), from Example 17.1's three points at t = 50, with the network's fixed LCG weights. Then the same steps with an oracle ε̂ = ε, which must recover x_0, land on the closed-form x_40 and reproduce the posterior mean of q(x_{t−1} | x_t, x_0).

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 3 rust      # from the repo root
cargo run -- -c 17 -e 3 python
cargo run --release -p c17e3  # the runner builds in debug mode; release is much faster
cargo test -p c17e3 --release     # the Rust parity test
pytest examples/chapter17/e03_ddpm_and_ddim_steps/python/main.py      # the Python parity test
```

## Parity

x_49 and x_40 match PyTorch to 1e-5; the three oracle checks agree to about 1e-7 (f32 rounding).
