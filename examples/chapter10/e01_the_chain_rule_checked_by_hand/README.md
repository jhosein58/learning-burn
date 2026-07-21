# Burn Autodiff: The Chain Rule (Example 10.1)

This repository contains a minimal example demonstrating backpropagation and the chain rule using the **Burn** deep learning framework in Rust, mapped directly against **PyTorch**.

## Overview

We verify the foundational math of automatic differentiation (Autodiff) using a simple, hand-calculated function:

$$u = a \cdot b$$
$$loss = v = u^2$$

For $a = 2.0$ and $b = 3.0$:
- **Forward Pass:** $u = 6.0$, $loss = 36.0$
- **Backward Pass (Gradients):** 
  - $\frac{\partial loss}{\partial a} = \frac{\partial loss}{\partial u} \cdot \frac{\partial u}{\partial a} = 2u \cdot b = 12 \cdot 3 = 36.0$
  - $\frac{\partial loss}{\partial b} = \frac{\partial loss}{\partial u} \cdot \frac{\partial u}{\partial b} = 2u \cdot a = 12 \cdot 2 = 24.0$

This example proves that Burn's autodiff engine yields the exact mathematical analytical derivatives.

## Key Concepts Covered

- **Manual vs. Automatic Differentiation:** Verifying symbolic math on paper before running the engine.
- **PyTorch to Burn Translation:** Direct conversion of dynamic graph backpropagation from Python to Rust.
- **Activation Memory:** Why the forward pass must store intermediate values (like $b$ to calculate $\frac{\partial u}{\partial a}$), explaining GPU/CPU memory scaling in deep learning.
