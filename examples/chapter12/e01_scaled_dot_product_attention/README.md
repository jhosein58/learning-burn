# Scaled Dot-Product Attention from Scratch (Example 12.1)

A minimal, transparent implementation of the core mathematical formula driving every modern Transformer architecture, using both **Burn** and **PyTorch**.

## The Core Formula
The entire mechanism is defined by a single, elegant equation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

This process operates in three distinct phases:
1. **Score:** Row-by-row dot products of Queries ($Q$) and Keys ($K$) measure similarity.
2. **Normalize:** A Softmax converts raw scores into weights that sum to $1$ along the key dimension.
3. **Average:** These weights compute a blended, weighted average of the Values ($V$).

## Why the Scaling Factor ($\sqrt{d_k}$) Matters
As the key dimension $d_k$ grows, dot product values naturally scale higher. Unscaled, large values push the softmax into saturation—meaning one key dominates, others drop to zero, and gradients vanish entirely. Dividing by $\sqrt{d_k}$ keeps the distribution soft and the gradients flowing during training.
