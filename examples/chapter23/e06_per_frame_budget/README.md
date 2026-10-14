# Example 23.6 — What One Frame Costs

The Example 23.5 detector priced per frame by formula (1,576 MACs, 1,697 parameters, 114 floats of state) and timed on a general-purpose CPU (our test machine) four ways: whole clip, recompute the window every frame, stream with Burn, stream with plain Rust slices. Timings describe the machine they run on; nothing here is measured on a microcontroller.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 23 -e 6 rust      # from the repo root
cargo run -- -c 23 -e 6 python
cargo test -p c23e6 --release     # the Rust parity test
pytest examples/chapter23/e06_per_frame_budget/python/main.py      # the Python parity test
```

## Parity

The budget numbers match in both languages, and the untrained model's first five logits match PyTorch to 1e-8. Burn's streaming step equals the whole clip exactly; the plain-slice step differs by ~1e-9 (a different summation order). The Python side imports the model from Example 23.5's `main.py`.
