# Example 16.4 — Training on Tiny Shakespeare

The real run: 8 epochs over the 1,406 shuffled training windows (704 steps of 16 × 128 tokens), AdamW, a 50-step warm-up to 1e-2 and a cosine down to 1e-3, loss / perplexity / accuracy metrics, the default checkpointers, and validation on the last 10% of the text after every epoch. It writes `config.json`, `vocab.txt` and `model.bpk` (burnpack) to `<temp dir>/learning-burn-c16/`, which Example 16.5 reads. It takes under two minutes on two cores.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference (same model, start, optimizer and schedule) + `pytest` test. |

## Run & test

```bash
cargo run --release -p c16e4   # from the repo root; a debug build is far slower
cargo run -- -c 16 -e 4 python
cargo test -p c16e4 --release     # the Rust test
pytest examples/chapter16/e04_train/python/main.py      # the Python test
```

## Parity

The two runs shuffle with different generators, so they are compared in size, not digit for digit (both end near a validation loss of 1.9). The tests check what is deterministic: the 704-step schedule against PyTorch's formula to 1e-12, the parameter count, and the loss of the first, unshuffled step.
