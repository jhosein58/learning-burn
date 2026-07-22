# Causal Masking: The Core of Autoregressive Transformers (Example 12.2)

A clean, illustrative example demonstrating how to restrict attention to past tokens, turning a bidirectional encoder into a generative autoregressive decoder (like GPT) using **Burn** and **PyTorch**.

## The Core Concept
For an autoregressive language model to predict the next token, it must not "cheat" by looking at future tokens. We enforce this constraint by applying an additive **lower-triangular mask** to the attention scores *before* they pass through the Softmax layer.

- **Allowed Connections:** We add `0.0` to the scores, leaving them unchanged.
- **Forbidden Connections (Future):** We add a huge negative value (such as `-1e9`).

Because $\exp(-10^9) \approx 0$, the softmax naturally assigns exactly $0.0$ weight to these future positions, cleanly eliminating information leakage.

## Key Takeaways
- **Why Add, Not Multiply?** Masking must happen before the softmax. Zeroing out weights after the softmax breaks the probability distribution because the remaining elements no longer sum to $1$.
- **Why $-1e9$ Instead of $-\infty$?** Real infinities in a tensor graph can easily multiply with zeros downstream to produce `NaN` values, ruining training. A large finite negative number underflows safely to zero without the stability risks.
