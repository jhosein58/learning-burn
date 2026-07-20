# Identity, Triangular Masks & Trace

PyTorch vs Burn: creating an identity with `eye`, extracting lower/upper triangles
with `tril`/`triu`, and summing the diagonal with `linalg::trace`. Also shows
batched matmul: `[8,2,3] × [8,3,4]` → `[8,2,4]`.

> The upper-triangular mask (`triu`) is the trick behind causal attention in
> transformers — zeroing out the future so a token cannot see what comes after it.
