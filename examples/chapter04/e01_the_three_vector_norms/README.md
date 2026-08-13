# Vector Norms: PyTorch vs. Burn (Rust)

This example compares how vector norms are computed in PyTorch and Burn, using the vector $x = [3, 4]$:

- **$L_2$ Norm (5.0):** Measures straight-line distance. In Burn, use `linalg::l2_norm`.
- **$L_1$ Norm (7.0):** Sums absolute values, driving weights to zero (Lasso). In Burn, use `linalg::l1_norm`.
- **$L_\infty$ Norm (4.0):** Takes the largest absolute component. In Burn, use `linalg::max_abs_norm`.

### Key Difference
While PyTorch configures different norms using the `ord` argument in a single function, Burn provides explicit, dedicated functions for each norm under the `linalg` module.
