# Example 15.3 — The Bigram as a Trainable Table

The same model as a 62x62 table of logits trained with softmax and cross-entropy. The gradient on the rows a batch uses is (p - one_hot) / N; over the whole text it can be written with the pair counts, (n_a softmax(W[a]) - C[a]) / M. Both forms are written by hand and checked against autodiff, then plain gradient descent (lr 100, 2000 full-data steps) drives the table to the counts of Example 15.2.

```rust
let n_a = c.clone().sum_dim(1);                        // [V, 1]
(softmax(w, 1) * n_a - c.clone()).div_scalar(m)        // dL/dW, all data
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

The text is `../data/shakespeare.txt`, embedded at compile time in Rust and read relative to the script in Python.

## Run & test

```bash
cargo run -- -c 15 -e 3 rust      # from the repo root
cargo run -- -c 15 -e 3 python
cargo test -p c15e3 --release     # the Rust parity test
pytest examples/chapter15/e03_bigram_gradient/python/main.py      # the Python parity test
```

## Parity

The hand gradients agree with Burn's autodiff and torch's autograd to ~1e-8; the loss after 500 steps matches torch to 1e-4, and both runs print the same lines.
