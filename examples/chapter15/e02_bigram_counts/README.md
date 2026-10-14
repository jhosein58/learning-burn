# Example 15.2 — A Bigram Model by Counting

Count how often each byte follows each other byte in the first 90% of the text, normalise every row, and the table is a language model. Its quality is the mean negative log-likelihood (NLL) of the held-out 10%, or its exponential, the perplexity. Add-alpha smoothing keeps pairs never seen in training from costing an infinite loss.

```rust
counts.clone().add_scalar(alpha) / rows.add_scalar(alpha * v) // add-alpha smoothing
let total: f32 = (eval.clone() * logp).sum().into_scalar();   // count-weighted NLL
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

The text is `../data/shakespeare.txt`, embedded at compile time in Rust and read relative to the script in Python.

## Run & test

```bash
cargo run -- -c 15 -e 2 rust      # from the repo root
cargo run -- -c 15 -e 2 python
cargo test -p c15e2 --release     # the Rust parity test
pytest examples/chapter15/e02_bigram_counts/python/main.py      # the Python parity test
```

## Parity

Burn and torch compute the same counts and the same NLLs (to 1e-5): train 2.3987 with the raw counts, validation 2.4820 (perplexity 11.96) with alpha 0.1, against 4.1271 (perplexity 62) for a uniform guess.
