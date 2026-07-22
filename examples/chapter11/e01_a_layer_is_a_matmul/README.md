# Burn From Scratch: What a Layer Actually Is (Example 11.1)

This repository contains a raw, from-scratch implementation of a multi-layer neural network using the **Burn** deep learning framework in Rust, alongside its **PyTorch** counterpart. 

## Overview

Strip away high-level abstractions like `nn::Linear` or `Module`, and a neural network is just a series of matrix multiplications (matmuls) with non-linear "bends" in between. Everything else is packaging.

In this example, we build a 2-layer network directly using raw tensors and core arithmetic operators:

1. **Input Batch:** $x$ $[2 \times 2]$ (2 examples, 2 features)
2. **Hidden Layer (Matmul):** $h_{pre} = x \mathbf{W_1}$ where $\mathbf{W_1}$ is $[2 \times 3]$
3. **Activation:** $h = \text{ReLU}(h_{pre})$ $[2 \times 3]$
4. **Output Layer (Matmul):** $out = h \mathbf{W_2}$ where $\mathbf{W_2}$ is $[3 \times 1]$

By bypassing high-level neural network modules, we expose the underlying shape matching and mathematical operations that define deep learning layers.

## Key Concepts Covered

- **Demystifying Layers:** Seeing firsthand that a "layer" is simply a matrix multiplication followed by an activation function.
- **The Shape Rule:** Tracking how dimensions change through the network ($[2 \times 2] \to [2 \times 3] \to [2 \times 1]$) to map inputs to predictions.
- **Floating-Point Realities:** Meeting $f32$ binary representation limitations (like $0.9$ representing as $0.90000004$) to understand why exact float equality checks ($==$) fail in practice.
- **No-Abstraction Burn Tensors:** Performing deterministic forward passes on `NdArray` backend without the standard boilerplate.
