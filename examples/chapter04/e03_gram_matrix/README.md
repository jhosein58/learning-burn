# Gram Matrix & Closed-Form Linear Regression

This example demonstrates how to compute the Gram matrix ($X^T X$) and the target projection ($X^T y$) to solve linear regression analytically using the normal equation:

$$(X^T X) w = X^T y$$

### Key Concepts & Differences
- **The Gram Matrix:** Summarizes any number of samples into a fixed $2 \times 2$ feature interaction matrix, bypassing iterative gradient descent.
- **PyTorch vs. Burn:** Both frameworks easily compute $X^T X$ and $X^T y$ using transpose and matrix multiplication (`matmul`). However, while PyTorch solves the system directly with `torch.linalg.solve`, Burn does not yet have a built-in solver and prints the intermediate matrices.
