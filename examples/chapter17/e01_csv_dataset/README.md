# Example 17.1 — A Dataset from a CSV File

`InMemDataset::from_csv` parses `data/machines.csv` into a serde struct, a `Standardize` batcher scales the three features with statistics fitted on the data, and a `DataLoader` batches 120 rows by 32.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 1 rust      # from the repo root
cargo run -- -c 17 -e 1 python
cargo test -p c17e1 --release     # the Rust parity test
pytest examples/chapter17/e01_csv_dataset/python/main.py      # the Python parity test
```

## Parity

Same 120 rows (42 failures), same means and standard deviations to 1e-5 (relative), same batch sizes `[32, 32, 32, 24]`, same first standardised row.
