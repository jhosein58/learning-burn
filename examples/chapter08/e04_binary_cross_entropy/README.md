# Binary Cross-Entropy (BCE) Loss

This example shows how to calculate Binary Cross-Entropy (BCE) loss, the default objective function for binary classification tasks (e.g., yes/no decisions).

### Key Concepts
- **Formulation:** Designed for single-probability outputs representing the likelihood of the positive class ("yes"). It collapses multiclass cross-entropy down to two mutually exclusive classes.
- **Sigmoid Fusion:** Similar to standard Cross-Entropy, BCE must be mathematically fused with the final activation function for numerical stability. 

### API Mapping & Pitfalls in Burn
- **Enabling Logits:** In PyTorch, using raw logits requires calling `F.binary_cross_entropy_with_logits`. In Burn, this is configured on the builder pattern by setting `.with_logits(true)` on `BinaryCrossEntropyLossConfig::new()`.
- **Target Formatting:** Targets in Burn are passed as an integer tensor containing binary class labels: `Tensor<B, 1, Int>`.
- **Avoid Manual Sigmoids:** Passing a manually sigmoid-activated tensor into a BCE loss initialized with `.with_logits(true)` will apply a double-sigmoid, dampening gradients and slowing down model training. Always feed raw, unbounded logits to a fused loss function.
