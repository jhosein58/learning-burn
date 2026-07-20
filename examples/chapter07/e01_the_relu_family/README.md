# Activation Functions: ReLU, Leaky ReLU, and GELU

This example compares the three most common hidden-layer activation functions ($ReLU$, $Leaky\ ReLU$, and $GELU$) in PyTorch and Burn.

### Key Concepts
- **ReLU ($max(0, x)$):** Flattens all negative values to zero. It is computationally the cheapest and is the default for most deep learning architectures.
- **Leaky ReLU:** Keeps a small fraction of negative values (e.g., slope of $0.1$). This prevents "dying neurons" by ensuring a small gradient still flows even when the input is negative.
- **GELU:** A smooth, probabilistic approximation used widely in Transformer models (like BERT and GPT) that trades computation speed for slightly better training performance.

### API Mapping
- PyTorch's `torch.nn.functional` namespace maps directly to Burn's `burn::tensor::activation` module.
- Both frameworks implement identical behaviors for `relu`, `leaky_relu`, and `gelu`.
