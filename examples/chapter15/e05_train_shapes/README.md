# Example 15.5 — Training the CNN

160 training images go through Burn's `Dataset` → `Batcher` → `DataLoader` in batches of 16. Training is 15 epochs of Adam (lr 0.02, eps 1e-8 to match PyTorch), and the test is 80 held-out images drawn with a different seed.

```rust
let loader = DataLoaderBuilder::<MyBackend, ShapeItem, ShapeBatch<MyBackend>>::new(ShapeBatcher)
    .batch_size(16)
    .build(train);
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 15 -e 5 rust      # from the repo root
cargo run -- -c 15 -e 5 python
cargo test -p c15e5 --release     # the Rust parity test
pytest examples/chapter15/e05_train_shapes/python/main.py      # the Python parity test
```

## Parity

The mean loss per epoch matches PyTorch to within 2e-3 (in practice to the 4th decimal), and both reach 80/80 on the test set. No shuffling here, so the batch order is identical: Chapter 17 covers shuffling.
