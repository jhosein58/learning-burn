# Maths, Clamp & Mask

PyTorch vs Burn: chaining `exp().log1p()` (softplus), clamping values with
`clamp(0.0, 1.0)`, and using a Bool tensor as a mask with `mask_where`.

Comparisons like `x > 0` produce a tensor of booleans, not a single `bool`.
`mask_where` uses that mask to replace values element-wise — no branching,
no `if`, fully parallel.

> A mask turns an "if" into a branch-free operation that runs efficiently on
> thousands of GPU cores.
