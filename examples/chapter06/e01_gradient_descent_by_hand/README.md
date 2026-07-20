# Gradient Descent by Hand

This example demonstrates a basic 4-step training loop (predict, measure error, compute gradient, step downhill) to fit a linear model ($y = 2x + 1$) without using automatic differentiation or built-in optimizers.

### Key Concepts & Differences
- **Manual Gradient Step:** Both frameworks compute the Mean Squared Error (MSE) gradient analytically using the formula: $\text{grad} = \frac{2}{n} X^T (Xw - y)$.
- **Explicit Cloning in Burn:** Because operations in Rust consume their inputs (ownership), tensors like `x` and `w` that are reused across loop iterations must be explicitly cloned using `.clone()`.
- **Convergence:** Starting from zero weights, the loop successfully converges the weight vector $w$ to approximately `[1.0, 2.0]` (representing the bias and slope).
