# Example 1.4 — Tensors Filled With Known Values

`zeros`, `ones`, `full`, `eye`, `arange` — the same constructors you know from
NumPy/PyTorch.

```rust
let z: Tensor<Backend, 2> = Tensor::zeros([2, 3], &device);
```

Note the style: annotate the binding and let it carry the type, rather than
turbofishing the call (`Tensor::<Backend, 2>::zeros(...)`).

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c1e4
cargo test --example c1e4
python python.py
pytest python.py
```

## Parity

Both sides assert: zeros/ones/full(7) over shape `[2, 3]`, a 3×3 identity, and
`arange(5..10) = [5, 6, 7, 8, 9]`.
