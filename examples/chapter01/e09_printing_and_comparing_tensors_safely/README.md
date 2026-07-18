# Printing & Comparing Tensors

Use standard Rust formatting to print tensors with the precision you want.

For floating-point tensors, compare values with `check_closeness()` instead of `==`.

> Floating-point values should be compared with a tolerance, not exact equality.
