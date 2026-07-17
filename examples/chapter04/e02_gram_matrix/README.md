# Example 4.2 — Gram Matrix

The building blocks of the normal equations for least squares: `X^T X` and
`X^T y`.

```rust
let xt = x.clone().transpose();
let gram = xt.clone().matmul(x);   // X^T X
let xty = xt.matmul(y);            // X^T y
```

Solving `gram · w = X^T y` for `w` needs `linalg::solve`, which isn't in Burn
0.21 yet (tracked upstream as burn#1538) — so this example stops at the two
products.

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent + `pytest` test. |

## Run & test

```bash
cargo run  --example c4e2
cargo test --example c4e2
python python.py
pytest python.py
```

## Parity

Both sides assert: `X^T X = [[3,6],[6,14]]`, `X^T y = [[6],[6]]`.
