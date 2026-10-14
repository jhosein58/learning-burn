# Example 18.4 — Fewer steps, and one round of reflow

Sweeps the number of Euler steps (1–32) for Example 18.2's model at w = 2, then makes 1,024 (noise, sample) pairs with the 16-step guided sampler and fine-tunes a copy on straight paths between them with the Learner (400 steps). The reflowed model is sampled at w = 1 with 1, 2 and 4 steps. Writes the 1-step grids to the system temp directory (`learning-burn-c18e4/`) and the reflowed model to `../weights/flow_unet_reflow.bpk`.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 18 -e 4 rust      # from the repo root (about a minute on 2 cores)
cargo run -- -c 18 -e 4 python
cargo test -p c18e4 --release     # the Rust test
pytest examples/chapter18/e04_few_steps_reflow/python/main.py
```

## Parity

64 pairs and the first three reflow losses match PyTorch to 1e-3.
