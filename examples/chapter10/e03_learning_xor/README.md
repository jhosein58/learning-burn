# Burn Neural Networks: Solving Non-Linear XOR (Example 10.3)

This repository contains a classic demonstration of backpropagation solving the **XOR problem** using the **Burn** deep learning framework in Rust, alongside its **PyTorch** counterpart.

## Overview

The XOR problem is the most historic argument for multi-layer neural networks. Because the four points of XOR are not linearly separable, a single linear layer (which only draws straight lines) can never solve it. 

By adding a hidden layer with a non-linear activation function ($\tanh$), the network can bend the input space to make the points separable. We train a simple Multi-Layer Perceptron (MLP):

$$\text{Input (2)} \to \text{Linear} \to \text{Tanh} \to \text{Linear (1)} \to \text{Output}$$

Using backpropagation, gradients travel through the non-linearity to optimize the first hidden layer, teaching the network to perfectly map the XOR truth table.

## Key Concepts Covered

- **The Power of Hidden Layers:** Why non-linearities and hidden units are essential for complex decision boundaries.
- **Optimization in Burn:** Defining custom structures with the `#[derive(Module)]` macro, initializing weights, and implementing full-batch Stochastic Gradient Descent (SGD) in Rust.
- **Early Convergence & Loss Limits:** Observing how loss converges to f32 numerical zero (around $10^{-12}$) and discussing why serious training pipelines benefit from *early stopping*.
- **Non-Convex Optimization:** Encountering local minima, showing that deep learning optimization relies on initialization and lacks the mathematical guarantees of simpler convex models.
