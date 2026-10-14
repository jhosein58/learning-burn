# Example 23.2 — A Causal Convolution, by Hand and One Frame at a Time

A causal 1-D convolution written as loops (left padding d(k-1) only), checked against Burn's functional `conv1d` with padding `(d(k-1), 0)`; then the streaming form, which keeps the last d(k-1) input frames in a ring buffer. A four-layer dilated stack (dilations 1, 2, 4, 8) gives the same 48 outputs whole and streamed.

```rust
let options = ConvOptions::new_with_padding([1], [(pad, 0)], [l.d], 1);
let yt = conv1d(xt, wt, None, options);
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 23 -e 2 rust      # from the repo root
cargo run -- -c 23 -e 2 python
cargo test -p c23e2 --release     # the Rust parity test
pytest examples/chapter23/e02_causal_conv_by_hand/python/main.py      # the Python parity test
```

## Parity

Small integers, so every sum is exact: loops, Burn's `conv1d`, the ring buffer, PyTorch's `F.pad` + `F.conv1d` and PyTorch's shift buffer agree to the last bit (`max|diff| = 0`). Changing input frame 10 leaves outputs 0..9 untouched.
