# Multi-Head Attention (Example 12.3)

A clean, step-by-step implementation of Multi-Head Attention (MHA) demonstrating how a model processes multiple types of relationships in parallel using **Burn** and **PyTorch**.

## The Core Concept
A single attention head can only focus on one type of relationship (e.g., subject-verb agreement). Multi-Head Attention solves this by splitting the feature dimension into several independent "heads," allowing each head to specialize in different aspects of the sequence (e.g., syntax, global summary, or local context) simultaneously.

## The 5-Step Pipeline
1. **Project:** Project input embeddings into Query ($Q$), Key ($K$), and Value ($V$) matrices using learned linear layers.
2. **Split:** Reshape and transpose the feature dimension to isolate the parallel heads:
   $$\text{[Sequence, Model Dimension]} \to \text{[Heads, Sequence, Head Dimension]}$$
3. **Attention:** Perform scaled dot-product attention in parallel across all heads using a single batched matrix multiplication (`matmul`).
4. **Merge (Glue):** Transpose and reshape back to recombine the heads:
   $$\text{[Heads, Sequence, Head Dimension]} \to \text{[Sequence, Model Dimension]}$$
5. **Output Projection:** Pass the concatenated representation through a final linear layer.

## Key Takeaway
Multi-Head Attention is fundamentally a shape manipulation. By using a leading "head" dimension, the actual attention math is processed efficiently in a single, parallelized batched operation without any slow loops.
