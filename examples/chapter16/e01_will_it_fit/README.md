# Example 16.1 — Will It Fit?

Chapter 14's feasibility check for a real model: every tensor of the 28.9M-parameter esp32-tinyLLM model, counted from its config and priced at f32 and at group-128 int4, against the ESP32-S3 N16R8's SRAM, PSRAM and flash. No framework.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 16 -e 1 rust      # from the repo root
cargo run -- -c 16 -e 1 python
cargo test -p c16e1 --release     # the Rust parity test
pytest examples/chapter16/e01_will_it_fit/python/main.py      # the Python parity test
```

## Parity

Same numbers in both languages: 28,869,920 parameters (87.2% in the PLE table), 115.5 MB in f32, and **14,912,332 bytes** in int4, which is exactly the size of the real `model.bin`.
