# Cosine Embedding Loss

This example shows how to calculate Cosine Embedding Loss, which evaluates the relationship between the directional angles of vector embeddings rather than their absolute magnitudes.

### Key Applications
- **Semantic Representation:** Used extensively to train text-embedding encoders (e.g., for vector search and Retrieval-Augmented Generation / RAG pipelines) by forcing semantically similar inputs to point in the same direction.
- **Verification Systems:** Powers biometric systems like face verification, where the task is to determine whether two distinct inputs represent the same entity, rather than performing multi-class classification.

### Key Concepts & Implementation
- **Direction over Magnitude:** The loss relies strictly on cosine similarity (the angle between vectors).
- **Target Indicators ($y \in \{1, -1\}$):**
  - $+1$: Indicates that vectors $a$ and $b$ should be highly similar (encouraging them to point in the same direction).
  - $-1$: Indicates that the vectors should be highly dissimilar (encouraging them to diverge).
- **API Mapping:** Instantiated in Burn using `CosineEmbeddingLossConfig::new().init()`. Inputs $a$ and $b$ are passed as 2D tensors of shape `[batch, embedding_dim]`, and targets $y$ are passed as a 1D integer tensor containing $1$ or $-1$ values.
