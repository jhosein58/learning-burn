# Reductions: Sum, Mean & Max

PyTorch vs Burn: reducing tensors with `sum`, `sum_dim`, `mean_dim`, and
`max_dim_with_indices`. The dim you name is the one that disappears.

Burn's `*_dim` methods keep the reduced axis (like PyTorch's `keepdim=True`),
which lets the result broadcast back against the original tensor.

> `max_dim_with_indices` returns both the values and the indices — the index
> of the largest logit is your model's predicted class.
