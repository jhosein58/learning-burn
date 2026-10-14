# Example 23.4 — A GRU, One Frame at a Time

A GRU step written from PyTorch's equations (gate order r, z, n; reset applied after the hidden matmul), checked against Burn's `nn::gru::Gru` over the whole sequence and one frame per call with the hidden state passed back in, and against `nn.GRU` with the same LCG weights. Also shows that `reset_after = false` is a different GRU.

```rust
let y = gru.forward(xt.clone().narrow(1, t, 1), state); // [1, 1, H]
let h_t = y.reshape([1, H]);
state = Some(h_t);
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 23 -e 4 rust      # from the repo root
cargo run -- -c 23 -e 4 python
cargo test -p c23e4 --release     # the Rust parity test
pytest examples/chapter23/e04_gru_streaming/python/main.py      # the Python parity test
```

## Parity

The last hidden state agrees with PyTorch to 1e-5 (in practice ~2e-7); whole-sequence and step-by-step Burn agree exactly.
