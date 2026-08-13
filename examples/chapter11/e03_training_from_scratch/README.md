# Burn From Scratch: Training with Zero Abstractions (Example 11.3)

This repository contains a raw, complete neural network training loop built using the **Burn** deep learning framework in Rust, alongside its **PyTorch** counterpart.

## Overview

We strip away every standard high-level abstraction: **no `nn::Linear` modules, no `Module` derives, and no optimizer configurations**. 

Instead, we train a Multi-Layer Perceptron to solve the classic non-linear XOR problem using just:
- Four raw weight and bias tensors ($\mathbf{W_1}$, $\mathbf{b_1}$, $\mathbf{W_2}$, $\mathbf{b_2}$) initialized with a uniform random distribution to break symmetry.
- A raw, four-beat loop: **forward pass** $\to$ **loss computation** $\to$ **backward pass** $\to$ **manual parameter update**.

Gradients are calculated automatically using Autodiff, but we apply the Gradient Descent step ourselves with a single line of tensor arithmetic per parameter:

$$\theta \leftarrow \theta - \eta \cdot \nabla_{\theta} L$$

## Key Concepts Covered

- **Zero-Optimizer Training:** Implementing the classic Stochastic Gradient Descent (SGD) update formula manually to realize that optimizers are just systematic loops over parameters.
- **Burn's Ownership & Graph Detachment:** Understanding why Burn does not use PyTorch's in-place mutation (like `with torch.no_grad(): W1 -= lr * grad`). Instead, we drop down to the plain backend tensor using `.inner()`, perform the update arithmetic, and lift it back into a fresh trainable leaf via `Tensor::from_inner(...).require_grad()`.
- **The Non-Existence of `zero_grad()`:** Discovering how Burn's functional ownership design naturally eliminates the need to manually zero out gradients; rebuilding weight leaves on each step means they start with clean slate histories.
- **Symmetry Breaking:** Why initializing weights with uniform random values is mathematically required to allow different hidden units to learn distinct features.
