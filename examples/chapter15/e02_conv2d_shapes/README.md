# Example 15.2 — 2D Convolution and the Shape Rule

A 3x3 vertical-edge kernel run over a 5x5 image finds the edge. Then the rule every conv layer obeys, `out = (n + 2p - d(k-1) - 1) / s + 1`, is checked against a real `nn::conv::Conv2d` for padding, stride and dilation.

```rust
let conv = Conv2dConfig::new([1, 8], [3, 3])
    .with_padding(PaddingConfig2d::Explicit(p, p, p, p))
    .with_stride([s, s])
    .with_dilation([d, d])
    .init::<Backend>(&device);
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 15 -e 2 rust      # from the repo root
cargo run -- -c 15 -e 2 python
cargo test -p c15e2 --release     # the Rust parity test
pytest examples/chapter15/e02_conv2d_shapes/python/main.py      # the Python parity test
```

## Parity

The edge map is `[3, 3, 0]` on every row in both frameworks. The four 12x12 cases give 10, 12, 6 and 8, the same as `nn.Conv2d`, and the weight shape is `[8, 1, 3, 3]` in both.
