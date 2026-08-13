# Gradient Checking: Verifying Backpropagation (Example 11.4)

A robust sanity check to prove that your backpropagation math and autodiff engine are completely correct.

## Why Use Gradient Checking?
Mathematical bugs in custom backward passes (such as an inverted transpose, a missing scaling factor, or a misplaced sign) rarely crash your program. Instead, the network simply trains poorly, leading to days of wasted troubleshooting. This check acts as a numerical unit test to catch those silent errors in seconds.

## How It Works
Instead of using calculus, we approximate the slope using only two forward passes (finite differences) by nudging a parameter slightly forward and backward:

$$f'(w) \approx \frac{f(w + \epsilon) - f(w - \epsilon)}{2\epsilon}$$

By comparing this numerical approximation against the gradient returned by **Burn's** Autodiff (or **PyTorch's** Autograd), we triangulate our results. When the numerical gradient and the analytical gradient match, your implementation is mathematically sound and trustworthy.
