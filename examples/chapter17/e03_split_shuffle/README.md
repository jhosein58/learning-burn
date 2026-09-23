# Example 17.3 — Splitting and Shuffling

A seeded 16/4 train/validation split with `ShuffledDataset` + `PartialDataset`, and a `DataLoader` with `.shuffle(seed)` that reorders the training set every epoch.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 3 rust      # from the repo root
cargo run -- -c 17 -e 3 python
cargo test -p c17e3 --release     # the Rust parity test
pytest examples/chapter17/e03_split_shuffle/python/main.py      # the Python parity test
```

## Parity

The two frameworks' random generators differ, so the *orders* differ. Both tests assert the same guarantees: disjoint split covering all items, every epoch a permutation of the training set, a new order each epoch, and the same order again from the same seed.
