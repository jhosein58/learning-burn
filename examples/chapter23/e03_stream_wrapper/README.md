# Example 23.3 — A Stream Wrapper for Conv1d

A Burn port of the idea behind Google's kws_streaming `Stream` layer (Rybakov et al., 2020): one trained `Conv1d`, the library's four modes (training, non-streaming inference, streaming with internal state, streaming with external state), and a buffer of B = max(0, d(k-1) + 1 - s) frames for kernel k, dilation d and stride s. Five (k, d, s) cases and a three-layer TCN, whole clip against streaming.

```rust
fn buffer_len(k: usize, d: usize, s: usize) -> usize {
    (d * (k - 1) + 1).saturating_sub(s)
}
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 23 -e 3 rust      # from the repo root
cargo run -- -c 23 -e 3 python
cargo test -p c23e3 --release     # the Rust parity test
pytest examples/chapter23/e03_stream_wrapper/python/main.py      # the Python parity test
```

## Parity

Weights and inputs are multiples of 1/32 and 1/8, so every mode gives identical numbers (`max|diff| = 0`) in both Burn and the PyTorch twin, and the first outputs match PyTorch to 1e-6.
