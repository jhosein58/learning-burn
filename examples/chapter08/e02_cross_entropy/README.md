# Cross-Entropy Loss

This example shows how to calculate the Cross-Entropy Loss, which is the standard loss function for classification tasks.

### Key Concepts
- **Focus on the Target:** Cross-entropy measures performance by focusing exclusively on the probability $p$ assigned to the correct target class, calculating $-\log(p)$. Output probabilities for incorrect classes do not directly enter the equation.
- **Asymmetric Penalty:** Because of the logarithmic curve, being confidently wrong is penalized severely. While a correct prediction with $p=0.9$ costs only $0.11$, an incorrect prediction with $p=0.05$ costs $3.0$ (nearly $30\times$ more), with the penalty approaching infinity as $p$ approaches $0$.
- **Fused Operation:** Both PyTorch and Burn apply $Softmax$ internally. Softmaxing your outputs manually before passing them to the loss function will result in a double-softmax, which flattens the distribution, blunts gradients, and degrades training without triggering a compiler error.
- **Numerical Stability:** Fusing the log-softmax and cross-entropy steps prevents underflow or $NaN$s. By canceling out adjacent exponential and logarithm operations algebraically, the computer avoids creating dangerously small float values.

### API Mapping & Implementation in Burn
- **Target Tensor Types:** The target indices must be passed as integer tensors (`Tensor<B, 1, Int>`).
- **Configuration Pattern:** Instead of instantiating the struct directly, Burn uses a config pattern: `CrossEntropyLossConfig::new().init(&device)`.
