# Choosing an Optimizer: SGD, Momentum, Adam, and AdamW

This example compares the strengths of modern optimizers and demonstrates how easily they can be swapped in Burn without modifying the rest of the training loop.

### Key Optimization Concepts
- **Momentum (SGD with Momentum):** Addresses the issue of oscillations (zig-zagging) in narrow loss valleys. By maintaining a running velocity of past gradients, opposing oscillations cancel out while consistent downhill directions accelerate progress.
- **Adam:** Introduces adaptive, per-parameter learning rates. It scales the step size for each weight based on the magnitude and variance of its historical gradients. This makes it highly robust and significantly easier to tune than plain SGD.
- **AdamW:** Corrects how weight decay ($L_2$ regularization) is handled in Adam. Instead of mathematically folding weight decay into the gradient updates (which distorts the adaptive step sizes), AdamW decouples it and applies the decay directly to the weights. It is the gold standard for training Transformer-based models.

### API Mapping & Swapability in Burn
- In Burn, changing optimizers requires swapping only the configuration line. The downstream four-beat training loop remains completely identical.
- **SGD with Momentum:** Configured via `SgdConfig::new().with_momentum(Some(MomentumConfig::new()))` (defaulting to a momentum factor of $0.9$).
- **Adam / AdamW:** Instantiated via `AdamConfig::new().init()` or `AdamWConfig::new().with_weight_decay(0.01).init()`.
- **Other Optimizers:** Burn also provides built-in support for algorithms like RMSprop, AdaGrad, and modern alternatives like Adan and Muon.
