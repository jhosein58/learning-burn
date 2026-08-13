# Standardising Features

PyTorch vs Burn: computing z-scores by subtracting the column mean and dividing
by the column standard deviation, all in a few lines. `mean_dim(0)` keeps the axis,
so the result broadcasts straight back over every row.

Burn's `var(dim)` divides by \(n-1\) (sample variance) — exactly what PyTorch does
by default. For population variance, use `var_bias(dim)`.

> Reductions preserve the reduced axis so the shape matches for broadcasting.
> The whole pipeline reads like the maths: \(z = (x - \mu) / \sigma\).
