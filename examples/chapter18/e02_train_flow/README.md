# Example 18.2 — Training by flow matching with the Learner

The U-Net of Example 18.1 trained with `SupervisedTraining` + `Learner` through `TrainStep`/`InferenceStep`, returning a `RegressionOutput`. The `Dataset` computes every item from its index: a fresh image, fresh noise, a time t and a label that is replaced by the "no label" class 10% of the time (for classifier-free guidance). AdamW, 1,200 steps of 64.

The trained model is written to `../weights/flow_unet.bpk` (burnpack, for Examples 18.3–18.5) and `../weights/flow_unet.safetensors` (PyTorch names and layout, via burn-store's `BurnToPyTorchAdapter`, for the Python sides), with the per-step loss log.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 18 -e 2 rust      # from the repo root (about 1.5 minutes on 2 cores)
cargo run -- -c 18 -e 2 python
cargo test -p c18e2 --release     # the Rust test
pytest examples/chapter18/e02_train_flow/python/main.py
```

## Parity

The first five batch losses match PyTorch's loop to 1e-3 (they agree to about 1e-5). The full PyTorch run prints the same per-200-step means and validation loss to four decimals.
