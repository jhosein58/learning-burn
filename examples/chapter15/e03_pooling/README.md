# Example 15.3 — Pooling and the Receptive Field

Max, average and global-average pooling on a 4x4 image of 1..16, plus how far one output pixel can "see" after conv → pool → conv.

```rust
let max = MaxPool2dConfig::new([2, 2]).init().forward(x.clone());
let avg = AvgPool2dConfig::new([2, 2]).init().forward(x.clone());
let global = AdaptiveAvgPool2dConfig::new([1, 1]).init().forward(x);
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 15 -e 3 rust      # from the repo root
cargo run -- -c 15 -e 3 python
cargo test -p c15e3 --release     # the Rust parity test
pytest examples/chapter15/e03_pooling/python/main.py      # the Python parity test
```

## Parity

Max `[6, 8, 14, 16]`, average `[3.5, 5.5, 11.5, 13.5]`, global `8.5`. The receptive field grows 3 → 4 → 8 pixels. All of it is identical in PyTorch.
