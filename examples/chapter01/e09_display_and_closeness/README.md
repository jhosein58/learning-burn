# Example 1.9 — Display, and Comparing Tensors Safely

Burn tensors use Rust's ordinary formatting (`{:.2}` rounds to two decimals), and
**never** compare floats with `==`.

```rust
let t = Tensor::<Backend, 2>::full([2, 3], 0.123456789, &device);
println!("{:.2}", t);
check_closeness(&a, &b);   // approximate comparison, with a report
```

## Files

| File | What it is |
|---|---|
| `rust.rs` | Burn version + `#[test]`. |
| `python.py` | PyTorch equivalent (`torch.allclose`) + `pytest` test. |

## Run & test

```bash
cargo run  --example c1e9
cargo test --example c1e9
python python.py
pytest python.py
```

## Parity

Both sides assert: `0.123456789` stored as f32 rounds to `~0.12345679`, and the
two nearly-equal vectors differ by `~0.001` (within `[1e-4, 1e-2)`).
`check_closeness` is Burn-specific and prints a report; PyTorch's analogue is
`torch.allclose`.
