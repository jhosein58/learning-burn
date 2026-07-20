# Autodiff Gradient Descent

This example shows how to replace manual calculus with Automatic Differentiation (autodiff) to calculate the Mean Squared Error (MSE) gradient for linear regression.

### Key Concepts & Differences
- **No Manual Formulas:** Instead of deriving the gradient by hand, the training loop relies entirely on `loss.backward()` to automatically apply the chain rule across all tensor operations.
- **Equivalent Outputs:** Starting at $w = [0, 0]$, both PyTorch and Burn compute the exact same loss ($41.0$) and gradient ($[-12.0, -35.0]$) as the manual formula from the previous example.
- **Implementation in Burn:** 
  - The backend is wrapped in `Autodiff<NdArray>`.
  - The weight tensor $w$ is marked with `.require_grad()`.
  - The gradient is safely retrieved from the returned container using `w.grad(&grads)`.
