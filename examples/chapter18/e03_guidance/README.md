# Example 18.3 — Classifier-free guidance, swept

Loads Example 18.2's model from `../weights/flow_unet.bpk`, samples 256 images (64 per class) with 16 Euler steps for guidance scales w = 0, 1, 2, 3, 4, 6, and scores each sample by its nearest clean shape (class and mean squared distance). Writes one PNG grid per w to the system temp directory (`learning-burn-c18e3/`).

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference (reads `../weights/flow_unet.safetensors`) + `pytest` test. |

## Run & test

```bash
cargo run -- -c 18 -e 3 rust      # from the repo root
cargo run -- -c 18 -e 3 python
cargo test -p c18e3 --release     # the Rust test
pytest examples/chapter18/e03_guidance/python/main.py
```

## Parity

Four guided samples (w = 2, 8 steps) from the same fixed noise match PyTorch to 1e-3; the full sweep prints the same table in both.
