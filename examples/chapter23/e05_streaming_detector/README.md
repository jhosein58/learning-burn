# Example 23.5 — Train on Whole Clips, Stream One Frame at a Time

A burst detector (three dilated causal `Conv1d`s, a `Gru`, a `Linear` head) trained for 96 Adam steps on whole clips of seeded noise with a sine burst, then run one frame per call with an external `State` (conv buffers + hidden state). Prints the test frame accuracy, the detection delay after a burst starts, and max|whole - streamed|.

```rust
let (logit, next) = model.step(x.clone().narrow(2, t, 1), state);
frames.push(logit);
state = next;
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 23 -e 5 rust      # from the repo root
cargo run -- -c 23 -e 5 python
cargo test -p c23e5 --release     # the Rust parity test
pytest examples/chapter23/e05_streaming_detector/python/main.py      # the Python parity test
```

## Parity

Same LCG data, same LCG init, same batch order: the first five losses match PyTorch to 1e-4, and the printed losses agree to the 4th decimal except the last (0.1242 against 0.1243); both reach 5916/6144 test frames. Streaming equals whole-clip inference to ~2e-6 in Burn and exactly in PyTorch.
