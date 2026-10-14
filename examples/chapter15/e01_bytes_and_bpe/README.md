# Example 15.1 — Bytes and BPE by Hand

Every text is already a sequence of bytes, so bytes are a 256-token vocabulary that can encode anything. Byte-pair encoding then merges the most frequent adjacent pair into a new token, again and again. We learn 48 merges on the Shakespeare slice, show the first 12, measure how much shorter the text gets, and check that encoding then decoding gives the sentence back.

```rust
let (pair, count) = most_frequent_pair(&ids); // ties go to the smallest pair
ids = merge(&ids, pair, 256 + k as u32);      // replace it, left to right
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

The text is `../data/shakespeare.txt`, embedded at compile time in Rust and read relative to the script in Python.

## Run & test

```bash
cargo run -- -c 15 -e 1 rust      # from the repo root
cargo run -- -c 15 -e 1 python
cargo test -p c15e1 --release     # the Rust parity test
pytest examples/chapter15/e01_bytes_and_bpe/python/main.py      # the Python parity test
```

## Parity

Rust and Python learn the same 48 merges in the same order (the tie-break is the same) and print identical output: 199,994 tokens as bytes, 138,684 after 48 merges, and an exact round trip.
