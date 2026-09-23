# Example 15.1 — 1D Convolution by Hand

A convolution is a small kernel sliding along a signal: multiply the overlap, add it up, move one step. The kernel `[1, 0, -1]` is a slope detector. We compute it with a plain loop, then with `burn::tensor::module::conv1d`, and get the same four numbers.

```rust
let x = Tensor::<Backend, 1>::from_floats(signal, &device).reshape([1, 1, 6]); // [batch, channels, length]
let w = Tensor::<Backend, 1>::from_floats(kernel, &device).reshape([1, 1, 3]); // [out, in, kernel]
let y = conv1d(x, w, None, ConvOptions::new([1], [0], [1], 1));
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 15 -e 1 rust      # from the repo root
cargo run -- -c 15 -e 1 python
cargo test -p c15e1 --release     # the Rust parity test
pytest examples/chapter15/e01_conv1d_by_hand/python/main.py      # the Python parity test
```

## Parity

The loop, `conv1d` and PyTorch's `F.conv1d` all give `[-3, -5, -7, -9]`. It is cross-correlation, the same as every deep-learning framework: the kernel is not flipped.
