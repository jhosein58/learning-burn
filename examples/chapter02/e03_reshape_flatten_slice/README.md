# Reshape, Flatten & Slice

PyTorch vs Burn: reshaping a flat tensor into a 2-D matrix, flattening it back,
slicing contiguous blocks, and gathering scattered rows with `select`.

Python's colon slices (`m[0:2, 1:3]`) become Rust ranges (`m.slice([0..2, 1..3])`).
For non-adjacent rows, use `select(dim, indices)`.

> Print the tensor directly (not `.to_data()`) to see the full shape, device, and dtype.
