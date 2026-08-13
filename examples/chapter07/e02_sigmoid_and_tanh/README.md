# Saturating Pair: Sigmoid and Tanh

This example demonstrates how to apply the classic saturating activation functions ($Sigmoid$ and $Tanh$) to squash input values into bounded ranges.

### Key Concepts & Differences
- **Sigmoid ($\sigma(x) \in (0, 1)$):** Commonly used for binary classification or probabilistic outputs. It is symmetric around $(0, 0.5)$, yielding exactly $0.5$ at $x = 0$.
- **Tanh ($\tanh(x) \in (-1, 1)$):** Outputs zero-centered values, yielding exactly $0.0$ at $x = 0$. 
- **Saturation:** Both functions flatten (saturate) at their extreme bounds. In these flat zones, the derivative approaches zero, leading to the vanishing gradient problem during backpropagation.

### API Mapping & Quirks in Burn
- **Sigmoid:** Only exists under the namespace `burn::tensor::activation::sigmoid(x)`. There is no `.sigmoid()` method on the tensor object.
- **Tanh:** Dual API existence. It can be called either as `burn::tensor::activation::tanh(x)` or as an element-wise tensor method `x.tanh()`. The latter is preferred in Burn for shorter syntax, matching operations like `.exp()` and `.sqrt()`.
