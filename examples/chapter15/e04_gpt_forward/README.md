# Example 15.4 — A Tiny GPT Forward Pass

A GPT built from raw `Tensor<2>` parameters in a `Vec`, without `burn::nn`: token and position embeddings, two pre-norm blocks of causal 4-head self-attention and a GELU MLP, LayerNorm by hand, and a head tied to the token embedding. 106,112 parameters from a fixed LCG. One forward pass over two 32-token windows of the text, and a check that the causal mask works: changing the last input token leaves every earlier position's logits exactly unchanged.

```rust
let tok = p[0].clone().select(0, idx);                 // embedding = row lookup
x = block(x, &p[2 + 12 * l..2 + 12 * (l + 1)], n, t);  // pre-norm block
x.matmul(p[0].clone().transpose())                      // tied head
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

The text is `../data/shakespeare.txt`, embedded at compile time in Rust and read relative to the script in Python.

## Run & test

```bash
cargo run -- -c 15 -e 4 rust      # from the repo root
cargo run -- -c 15 -e 4 python
cargo test -p c15e4 --release     # the Rust parity test
pytest examples/chapter15/e04_gpt_forward/python/main.py      # the Python parity test
```

## Parity

The logits and the loss (4.323093, against ln 62 = 4.127134) match PyTorch to 1e-5.
