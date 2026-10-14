# Example 15.5 — Training the Tiny GPT with a Hand-written Adam

Example 15.4's GPT trained on the Shakespeare slice for 2000 steps (batch 16 x 32) with Adam written out: first and second moments, bias correction, epsilon 1e-8, learning rate 3e-3. Autodiff computes the gradients; the update runs off autodiff and puts each parameter back as a fresh leaf. Then 200 characters sampled at temperature 0.8. One to two minutes on two CPU cores.

```rust
let m = m.mul_scalar(BETA1) + g.clone().mul_scalar(1.0 - BETA1);
let update = m_hat / v_hat.sqrt().add_scalar(EPS);
let new = p.inner() - update.mul_scalar(LR);
out.push(Tensor::from_inner(new).require_grad());
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

The text is `../data/shakespeare.txt`, embedded at compile time in Rust and read relative to the script in Python.

## Run & test

```bash
cargo run -- -c 15 -e 5 rust      # from the repo root
cargo run -- -c 15 -e 5 python
cargo test -p c15e5 --release     # the Rust parity test
pytest examples/chapter15/e05_gpt_training/python/main.py      # the Python parity test
```

## Parity

Same LCG weights, same LCG batches: the first three losses match PyTorch to 1e-4 (the test), and on our machine the whole run, sample included, printed the same lines in both languages. Validation loss falls from 4.3167 to 1.8993, below the bigram's 2.48.
