# Example 23.1 — Frames, Hops and the Price of Forgetting

A 16 kHz stream cut into 25 ms frames every 10 ms, and a four-layer dilated causal conv stack (40 features in, 64 channels, k = 3, dilations 1, 2, 4, 8) priced per hop three ways: recompute the whole 98-frame window, recompute only the receptive-field cone, or keep state and stream. No framework on the Rust side; the PyTorch side also measures the receptive field with autograd.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]` (no dependencies). |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 23 -e 1 rust      # from the repo root
cargo run -- -c 23 -e 1 python
cargo test -p c23e1 --release     # the Rust parity test
pytest examples/chapter23/e01_frames_and_budget/python/main.py      # the Python parity test
```

## Parity

The Rust test and the pytest test check the same integers: 98 frames, a 31-frame receptive field, 44,544 MACs per hop streaming against 751,104 (cone) and 4,365,312 (window), 1,872 floats of state. Autograd confirms that the newest output depends on exactly 31 input frames.
