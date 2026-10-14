# Example 15.7 — A KV Cache by Hand

Example 15.4's GPT with a 128-token context generates 116 tokens greedily twice: once by re-running the whole prefix at every step, once with a cache of every block's keys and values, feeding each token only once. Every matmul goes through a counter of multiply-adds.

```rust
let k = append(&mut kv.k[l], qkv.clone().slice([0..1, D..2 * D])); // [t, D]
let scores = mm(q, k.swap_dims(1, 2)).mul_scalar(scale); // [H, 1, t]
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

The text is `../data/shakespeare.txt`, embedded at compile time in Rust and read relative to the script in Python.

## Run & test

```bash
cargo run -- -c 15 -e 7 rust      # from the repo root
cargo run -- -c 15 -e 7 python
cargo test -p c15e7 --release     # the Rust parity test
pytest examples/chapter15/e07_kv_cache/python/main.py      # the Python parity test
```

## Parity

Both ways choose the same tokens, with logits equal to ~6e-7; the cache cuts the multiply-adds from 1,001,252,608 to 15,069,312 (66.4x). The counts, tokens and final logits match PyTorch. Wall-clock times are printed but not tested.
