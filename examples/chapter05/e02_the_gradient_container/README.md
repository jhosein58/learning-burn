# Gradient Storage: PyTorch vs. Burn (Rust)

This example highlights the most significant architectural difference between PyTorch and Burn regarding where gradients are stored, using $L = \sum(a \cdot b)$ (where $\frac{\partial L}{\partial a} = b$ and $\frac{\partial L}{\partial b} = a$).

### The Core Difference
- **PyTorch (Side Effects):** Mutates tensors in-place. Calling `loss.backward()` returns nothing and attaches the gradients directly to the tensors as an attribute (`a.grad`).
- **Burn (Pure & Immutable):** Avoids hidden mutation to align with Rust's ownership model. Calling `loss.backward()` returns an independent `grads` container. Tensors remain completely untouched, and gradients must be looked up explicitly via `a.grad(&grads)`.

### Common Pitfall: `.unwrap()`
In Burn, `tensor.grad(&grads)` returns an `Option`. Calling `.unwrap()` will panic if:
1. You forgot to call `.require_grad()` on the tensor.
2. The tensor was not used in the forward pass to compute the loss.
