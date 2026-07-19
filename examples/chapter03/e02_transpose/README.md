# Transpose

PyTorch vs Burn: flipping a matrix over its diagonal with `.t()` or `.transpose()`.
Burn's transpose consumes the tensor, so reuse requires `.clone()` first.

You reach for transpose most often to fix a shape mismatch before a matmul —
the inner dimensions must align.

> Transpose reflects across the main diagonal: shape `[2, 3]` becomes `[3, 2]`.
> The operation is central to backpropagation, which is a chain of transposed
> matrix products.
