# Example 17.5 — Flow Matching by Hand

The straight path x_t = (1−t) x_0 + t x_1 from noise x_0 to data x_1 and its velocity x_1 − x_0. The same MLP and backward pass as Example 17.2, now with the flow-matching loss L = (1/N) Σ‖v̂ − (x_1 − x_0)‖² on the first training batch of Example 17.6. Then one Euler step and one Heun step (t = 0.5 → 0.6) with fixed inputs, and the oracle check that both stay exactly on a straight path.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 5 rust      # from the repo root
cargo run -- -c 17 -e 5 python
cargo run --release -p c17e5  # the runner builds in debug mode; release is much faster
cargo test -p c17e5 --release     # the Rust parity test
pytest examples/chapter17/e05_flow_matching_by_hand/python/main.py      # the Python parity test
```

## Parity

The hand-written gradients equal Burn's autodiff exactly; the loss, gradients and both ODE steps match PyTorch to 1e-5.
