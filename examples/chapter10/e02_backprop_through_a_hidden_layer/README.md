# Burn Autodiff: Multi-Layer Backpropagation (Example 10.2)

This repository contains a minimal example demonstrating backpropagation through a hidden layer using the **Burn** deep learning framework in Rust, alongside its **PyTorch** counterpart. 

## Overview

Unlike a single chain, real backpropagation involves propagating gradients backward through multiple layers to reach the earliest weights. This example builds a tiny 2-layer neural network with fixed, reproducible weights to show how gradients travel through non-linearities:

$$\mathbf{h_{pre}} = \mathbf{x} \mathbf{W_1}$$
$$\mathbf{h} = \text{ReLU}(\mathbf{h_{pre}})$$
$$\mathbf{out} = \mathbf{h} \mathbf{W_2}$$
$$\text{loss} = \sum (\mathbf{out} - \mathbf{target})^2$$

By utilizing deterministic inputs and weights, we can hand-check the exact values flowing forward and backward through the hidden layer.

## Key Concepts Covered

- **Deep Backpropagation:** How a single `backward()` call populates gradients for both the near layer ($\mathbf{W_2}$) and the deep layer ($\mathbf{W_1}$).
- **The Role of ReLU:** Both activations in this example are positive ($0.7$ and $1.0$), letting the gradient pass through cleanly. If they were negative, the derivative of ReLU would act as a gate, multiplying the gradient by zero and causing it to vanish.
- **PyTorch Equivalency:** Direct translation of multi-layer linear and activation passes from Python into idiomatic Rust using Burn's Tensor API.
