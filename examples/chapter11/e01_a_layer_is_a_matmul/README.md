# Burn From Scratch: What a Layer Actually Is (Example 11.1)

This repository contains a raw, from-scratch implementation of a multi-layer neural network using the **Burn** deep learning framework in Rust, alongside its **PyTorch** counterpart. 

## Overview

Strip away high-level abstractions like `nn::Linear` or `Module`, and a neural network is just a series of matrix multiplications (matmuls) with non-linear "bends" in between. Everything else is packaging.

In this example, we build a 2-layer network directly using raw tensors and core arithmetic operators:

1. **Input Batch:** $x$ $[2 \times 2]$ (2 examples, 2 features)
2. **Hidden Layer (Matmul):** $h_{pre} = x \mathbf{W_1}$ where $\mathbf{W_1}$ is $[2 \times 3]$
3. **Activation:** $h = \text{ReLU}(h_{pre})$ $[2 \times 3]$
4. **Output Layer (Matmul):** $out = h \mathbf{W_2}$ where $\mathbf{W_2}$ is $[3 \times 1]$
5. **Loss (MSE):** $\mathcal{L} = \frac{1}{N} \sum (out - t)^2$, a single scalar scoring the batch

By bypassing high-level neural network modules, we expose the underlying shape matching and mathematical operations that define deep learning layers.

## The Loss

A forward pass alone tells you what the network *predicts*; the loss tells you how *wrong* it is. We score the predictions against targets $t$ $[2 \times 1]$ with mean squared error: subtract, square, average.

Two details are worth pausing on, because both are easy to get wrong:

- **The target must be shaped $[2 \times 1]$, not $[2 \times 2]$.** The network emits one number per example, so there is one target per example. Give it a $[2 \times 2]$ target and neither Burn nor PyTorch complains — they *broadcast*, quietly comparing every prediction against every target column and handing back 4 numbers instead of 2. Silent broadcasting is one of the most common sources of a wrong-but-plausible loss.
- **The loss must reduce to a scalar.** Squaring the errors leaves a $[2 \times 1]$ tensor of per-example penalties; `.mean()` collapses it into the single number that backprop will later differentiate.

With the fixed weights in this example:

```
out    = [[2.94], [6.38]]
t = [[1.00], [0.00]]
sq_err = [[3.7636], [40.7044]]
loss   = 22.234
```

Nothing here trains yet — no gradients, no updates. That comes in Example 11.2, which reuses this exact network, targets, and loss, then differentiates it by hand and by autodiff. The $22.234$ you see above is the same number that example starts from.

## Key Concepts Covered

- **Demystifying Layers:** Seeing firsthand that a "layer" is simply a matrix multiplication followed by an activation function.
- **The Shape Rule:** Tracking how dimensions change through the network ($[2 \times 2] \to [2 \times 3] \to [2 \times 1]$) to map inputs to predictions.
- **Scoring a Prediction:** Building mean squared error from scratch — subtract, square, average — instead of reaching for a loss module.
- **Broadcasting Traps:** Learning why a mis-shaped target produces a loss that runs cleanly and is still wrong.
- **Floating-Point Realities:** Meeting $f32$ binary representation limitations (like $0.9$ representing as $0.90000004$) to understand why exact float equality checks ($==$) fail in practice.
- **No-Abstraction Burn Tensors:** Performing deterministic forward passes on `NdArray` backend without the standard boilerplate.
