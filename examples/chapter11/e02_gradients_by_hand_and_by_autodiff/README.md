# Burn Autodiff Demystified: Gradients Computed Twice (Example 11.2)

This repository contains an essential, eye-opening comparison of backpropagation implemented **by hand** versus **automatic differentiation (Autodiff)** using the **Burn** deep learning framework in Rust, alongside its **PyTorch** counterpart.

## Overview

Backpropagation is often treated like magic, but it is just a sequence of matrix multiplications with transposes and element-wise masks. In this example, we calculate the gradients of a 2-layer neural network using two completely different methods:

1. **Analytical Backprop (By Hand):** We write out the exact mathematical chain rule recipe using manual tensor transposes, matrix multiplications, and a ReLU derivative mask.
2. **Automatic Differentiation (Autodiff):** We call Burn's `loss.backward()` and extract the engine's computed gradients.

By running both, we demonstrate that the manual math and the engine's automated path yield **digit-for-digit identical results**—down to the exact floating-point noise in the last decimal place.

## The Backpropagation Recipe

Starting from the Mean Squared Error (MSE) loss and moving backward through the network:

$$\mathbf{dout} = \frac{2}{N} (\mathbf{out} - \mathbf{target})$$
$$\mathbf{dW_2} = \mathbf{h^T} \mathbf{dout}$$
$$\mathbf{dh} = \mathbf{dout} \mathbf{W_2^T}$$
$$\mathbf{dh_{pre}} = \mathbf{dh} \odot \mathbb{I}(\mathbf{h_{pre}} > 0) \quad \text{(ReLU Mask)}$$
$$\mathbf{dW_1} = \mathbf{x^T} \mathbf{dh_{pre}}$$

*Note: In Burn, the ReLU derivative is cleanly implemented using a binary mask: `.mask_fill(h_pre.lower_equal_elem(0.0), 0.0)`.*

## Key Concepts Covered

- **Demystifying Autodiff:** Proving that `backward()` is not a black box; it executes the exact same chain rule multiplications you would write on paper, just faster and without human error.
- **Transposes and Dimensions:** Showing how transposing matrices ($\mathbf{x^T}$, $\mathbf{h^T}$, $\mathbf{W_2^T}$) is mathematically required to align shapes during gradient flow.
- **The ReLU Gate:** Zeroing out gradients where pre-activations were non-positive, illustrating the dead/live unit mechanism in backpropagation.
