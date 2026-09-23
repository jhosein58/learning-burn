# Example 17.4 — Augmentation as a Lazy Mapping

A `Mapper` enum (identity / mirror left-right) applied through `MapperDataset`, and the two halves joined with `ComposedDataset`: 24 images become 48, nothing stored twice.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 4 rust      # from the repo root
cargo run -- -c 17 -e 4 python
cargo test -p c17e4 --release     # the Rust parity test
pytest examples/chapter17/e04_augment/python/main.py      # the Python parity test
```

## Parity

Same bright row before and after mirroring, byte for byte, and mirroring twice returns the original with its label.
