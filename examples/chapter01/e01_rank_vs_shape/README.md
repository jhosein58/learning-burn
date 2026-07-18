# Example 1.1 — Rank Is Not Shape

This example demonstrates one of the most common beginner mistakes in Burn:

> **The rank of a tensor is NOT the number of elements it contains.**

## What this example shows

A tensor containing five numbers is still a **rank-1** tensor because it has only **one axis**.

```text
[1.0, 2.0, 3.0, 4.0, 5.0]
```

Correct:

```rust
Tensor::<Backend, 1>::from_floats(...)
```

Incorrect:

```rust
Tensor::<Backend, 5>::from_floats(...)
```

`Tensor<Backend, 5>` means a **5-dimensional tensor**, not a tensor with five values.

## Key takeaway

- **Rank** = number of axes (compile-time)
- **Shape** = size of each axis (runtime)

A tensor with shape `[5]` has:

- Rank: **1**
- Shape: **[5]**

These are different concepts, and confusing them is one of the most common mistakes when learning tensor libraries.

## Files

| File | What it is |
|---|---|
| `rust.rs` | The Burn version, plus a `#[test]` that checks the result. |

## Run it

```bash
cargo run --example c1e1        # Burn
```

## Verify parity

Both sides assert the **same golden values** — rank 1, shape `[5]`, values
`[1, 2, 3, 4, 5]` — so if either framework drifts, its test fails.

```bash
cargo test --example c1e1       # Burn test (note: --example, not plain `cargo test`)
```
