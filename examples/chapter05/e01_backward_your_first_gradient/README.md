# Autodiff: Computing Your First Gradient

This example demonstrates the basics of Automatic Differentiation (autodiff) in PyTorch and Burn, verifying the machine's calculus using the function $f(x) = \sum x^2$ (where the derivative $\frac{\partial f}{\partial x} = 2x$) for $x = [1, 2, 3]$.

### Key Concepts & Differences
- **Tracking Gradients:** PyTorch uses `requires_grad=True` on tensor creation, while Burn uses the `.require_grad()` method.
- **Backend Decorators:** In Burn, autodiff is enabled by wrapping a base backend (like `NdArray`) with the `Autodiff` decorator (`Autodiff<NdArray>`).
- **Gradient Retrieval:** In PyTorch, calling `.backward()` populates `x.grad` in-place. In Burn, `f.backward()` returns a `grads` container, from which the gradient is explicitly extracted via `x.grad(&grads)`.
- **Scalar Requirement:** Both frameworks require the backward pass to start from a single scalar value (hence the `.sum()`), as gradients can only be computed with respect to a single loss value.
