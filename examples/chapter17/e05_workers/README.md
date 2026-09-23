# Example 17.5 — Keeping the Model Fed

A deliberately slow dataset (4 ms per item) loaded with 0, 2 and 4 workers. Burn's workers are threads over one shared dataset; PyTorch's are processes.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 5 rust      # from the repo root
cargo run -- -c 17 -e 5 python
cargo test -p c17e5 --release     # the Rust parity test
pytest examples/chapter17/e05_workers/python/main.py      # the Python parity test
```

## Parity

Timings are machine-dependent and not asserted. Both tests assert that every one of the 64 items arrives exactly once, whatever the number of workers.
