# Example 16.2 — A GPT as a Burn Module

`GptConfig` builds a character-level GPT from `burn::nn`: token and learned position `Embedding`s, a `TransformerEncoder` with `norm_first = true` and the causal mask of `generate_autoregressive_mask`, a final `LayerNorm` and a bias-free `Linear` head. With the defaults (`T = 128`, `d = 64`, 4 heads, 2 layers, 62 symbols) it has 116,224 parameters. The weights come from a shared LCG and are copied into PyTorch's `nn.TransformerEncoder` (packed `in_proj_weight`, and `norm1`/`norm2` swapped against Burn's `norm_2`/`norm_1`).

```rust
let causal = generate_autoregressive_mask(n, t, &device); // [N, T, T], true = hidden
let x = self.blocks.forward(TransformerEncoderInput::new(x).mask_attn(causal));
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 16 -e 2 rust      # from the repo root
cargo run -- -c 16 -e 2 python
cargo test -p c16e2 --release     # the Rust parity test
pytest examples/chapter16/e02_gpt_module/python/main.py      # the Python parity test
```

## Parity

On the first two training windows, the logits and the cross-entropy (4.3624) agree with PyTorch to 1e-4 (in practice to about 3e-7), both count 116,224 parameters, and changing the last input token leaves the logits at every earlier position exactly unchanged.
