# Example 16.5 — Generating Text

A separate program that knows nothing about training: it reads `config.json`, `vocab.txt` and `model.bpk` from Example 16.4's directory, rebuilds the GPT and samples from it — greedy, and with temperature and top-k, all driven by the shared LCG's uniform numbers. It also generates through Burn's `TransformerEncoder` autoregressive cache and checks that the tokens are identical.

```rust
let (top, ids) = logits.div_scalar(temperature).topk_with_indices(top_k, 0);
let cdf = softmax(top, 0).cumsum(0);
let below = cdf.lower_scalar(u).int().sum().into_scalar::<i64>() as usize;
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test (main() needs Example 16.4's `python/main.py` run first). |

## Run & test

```bash
cargo run --release -p c16e4   # train first
cargo run --release -p c16e5
cargo run -- -c 16 -e 5 python
cargo test -p c16e5 --release     # the Rust parity test
pytest examples/chapter16/e05_generate/python/main.py      # the Python parity test
```

## Parity

On the untrained LCG model of Example 16.2, with the same uniform numbers, both frameworks draw the same 24 characters after `"First"` at temperature 0.8 and top-k 10, and the cached generation draws them too.
