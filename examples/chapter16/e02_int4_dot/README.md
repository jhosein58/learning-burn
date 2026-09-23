# Example 16.2 — The int4 Dot Product

The firmware's hot loop at eight-element scale: weights stored as value + 8 in 4-bit nibbles, two per byte, multiplied by i16 activations, with the `- 8` hoisted out of the loop as `8 * sum(activations)`.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 16 -e 2 rust      # from the repo root
cargo run -- -c 16 -e 2 python
cargo test -p c16e2 --release     # the Rust parity test
pytest examples/chapter16/e02_int4_dot/python/main.py      # the Python parity test
```

## Parity

Packed bytes `0x6B 0x0F 0xD8 0xC7` and dot product `142` in both, naive and hoisted, as exact integers.
