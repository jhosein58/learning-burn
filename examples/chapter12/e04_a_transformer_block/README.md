# A Transformer Block from Scratch (Example 12.4)

An assembly of a complete Transformer Encoder block from scratch using **Burn** and **PyTorch**, integrating self-attention, feed-forward networks, residual connections, and custom LayerNorm.

## The Core Concept
A standard Transformer block consists of two main sub-layers that process token sequences:
1. **Self-Attention Sub-Layer:** Mixes information *across* tokens (the only place where tokens communicate).
2. **Feed-Forward Sub-Layer:** Processes *each token independently* using a Multi-Layer Perceptron (MLP).

These sub-layers are wrapped in **Residual Connections** and **Layer Normalization** to keep activations and gradient flow stable.

## Key Architecture Details
- **Residual Connections (`x + SubLayer(x)`):** Prevents vanishing gradients. The derivative contains a $+1$ term, providing a highway for gradients to flow backward unchanged through hundreds of layers.
- **Layer Normalization (LayerNorm):** Rescales the features of each token individually to have zero mean and unit variance. Unlike BatchNorm, it operates independently of batch size, making it perfect for varying sequence lengths and single-batch inference.
- **Shape Preservation:** The block outputs a tensor of the exact same shape as the input (e.g., `[Sequence, Model Dimension]`), allowing multiple blocks to be stacked infinitely.
