# Disabling Gradient Tracking (Inference)

This example shows how to perform cheap, memory-efficient inference by disabling gradient tracking in both PyTorch and Burn.

### Key Concepts & Differences
- **No Global State Blocks:** Unlike PyTorch, which relies on a dynamic global context manager (`with torch.no_grad():`), Burn has no special state blocks to enter and leave.
- **Peeling the Autodiff Wrapper (`.inner()`):** In Burn, autodiff is a type decorator (`Autodiff<Backend>`). To stop tracking gradients, you simply call `.inner()` on a tensor. This peels away the autodiff wrapper and hands you the plain backend tensor (e.g., `NdArray`), which naturally performs no graph recording.
- **Type-Safe Inference:** Whether gradients are tracked is written directly into the Rust types (`Tensor<Autodiff<NdArray>>` vs `Tensor<NdArray>`). The compiler ensures you never accidentally perform tracked operations during inference.
