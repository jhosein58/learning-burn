# Ownership & Cloning

Burn follows Rust's ownership rules. Most tensor operations consume their inputs.

This example demonstrates why `.clone()` is often needed when reusing the same tensor.

> Cloning a tensor is cheap—it clones the handle, not the underlying data.
