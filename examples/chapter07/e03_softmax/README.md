# Softmax — Turning Scores into Probabilities

This example demonstrates how to use the $Softmax$ function to transform raw, unbounded logits into a normalized probability distribution.

### Key Concepts
- **Probability Distribution:** $Softmax$ exponentiates each input logit and divides it by the sum of all exponentiated logits. This forces the outputs to reside in the $(0, 1)$ range and guarantees they sum to exactly $1.0$.
- **Exponential Amplification:** Because it relies on $exp(x)$, modest differences in logits (e.g., $[1, 2, 3]$) translate to large differences in probability (e.g., $[9\%, 24\%, 67\%]$).
- **Numerical Stability:** To prevent overflow ($exp(1000) = \infty \to NaN$), Burn's internal implementation subtracts the maximum logit value from each element before exponentiating, ensuring mathematical equivalence without numerical instability.

### API Mapping & Pitfalls
- **Dimension Argument:** PyTorch's keyword argument `dim=1` maps directly to a positional index in Burn: `activation::softmax(logits, 1)`. 
- **Axis Selection:** For a standard `[batch, classes]` tensor, the normalization dimension must be the class axis (`1`). Normalizing over axis `0` mistakenly normalizes across the batch, which compiles without errors but ruins model training.
