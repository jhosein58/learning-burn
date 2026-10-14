# Example 16.3 — Sixteen Learner Steps, Checked

Example 16.2's GPT behind `TrainStep` / `InferenceStep`, with a cross-entropy over the `[N·T, V]` logits and a `SequenceOutput` for the metrics, trained by the `Learner` with AdamW and a linear warm-up + cosine schedule (`SequentialLrSchedulerConfig`). The run is short and deterministic: the first 128 training windows, in order, for two epochs (16 steps), from the LCG weights. The per-step learning rate and loss are read back from the Learner's `Learning_Rate.log` and `Loss.log`.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference (the loop the Learner runs, written out) + `pytest` test. |

## Run & test

```bash
cargo run --release -p c16e3   # from the repo root; a debug build is far slower
cargo run -- -c 16 -e 3 python
cargo test -p c16e3 --release     # the Rust parity test
pytest examples/chapter16/e03_learner_steps/python/main.py      # the Python parity test
```

## Parity

All sixteen learning rates agree with `torch.optim.AdamW` driven by the same schedule to 1e-9, and all sixteen losses to 1e-3 (in practice to about 1e-6); the validation loss after each epoch agrees to 1e-3. Both sides use `eps = 1e-8` and `weight_decay = 0.01` (PyTorch's defaults; Burn's are 1e-5 and 1e-4).
