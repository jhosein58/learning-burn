# Example 4.1 — Vector Norm, Normalize, Determinant

```rust
let n2 = linalg::l2_norm(x.clone(), 0);              // 5.0
let unit = linalg::vector_normalize(x, 2.0, 0, 1e-12); // [0.6, 0.8]
let d = linalg::det::<Backend, 3, 2, 1>(m);          // -6.0
```

## The `det` batch-rank rule

`det` operates on a **batch** of square matrices, so its input rank must be ≥ 3:
`[1, 2, 2]` is a batch of one 2×2 matrix. A plain `[2, 2]` compiles but **panics
at runtime** (`TensorCheck` requires rank ≥ 3). The extra generics follow from
the rank: `D1 = D-1`, `D2 = D-2`.

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent (`torch.linalg.norm`/`det`) + `pytest` test. |

## Run & test

```bash
cargo run  --example c4e1
cargo test --example c4e1
python python.py
pytest python.py
```

## Parity

Both sides assert: L2 norm `5`, unit vector `[0.6, 0.8]`, determinant `-6`.
