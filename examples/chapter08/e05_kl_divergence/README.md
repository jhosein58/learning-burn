# Kullback-Leibler (KL) Divergence Loss

This example shows how to calculate Kullback-Leibler (KL) divergence, which measures the difference between two probability distributions rather than comparing a prediction to a single categorical label.

### Key Applications
- **Knowledge Distillation:** A small student model is trained using KL divergence to mimic the complete, softened output probability distribution of a larger teacher model (transferring rich relational signals, such as close synonyms, instead of just the top-one label).
- **RLHF Alignment Leash:** During Reinforcement Learning from Human Feedback, KL divergence is used as a penalty term to measure and restrict how far the updated model policy drifts from the original base model, preventing reward hacking and linguistic collapse.

### API Mapping & Critical Gotchas in Burn
- **Input Asymmetry:** Like PyTorch, Burn's KL divergence loss is highly asymmetric regarding its inputs:
  - **Prediction (Student):** Must be formatted as **log-probabilities** (using `log_softmax`).
  - **Target (Teacher):** Must be formatted as **standard probabilities** (using `softmax`).
  - *Swapping these or failing to apply log-transformations to the prediction will produce incorrect losses without compiling or runtime errors.*
- **Reduction mapping:** PyTorch's mathematically correct `reduction="batchmean"` maps directly to `Reduction::Auto` in Burn.
- **Instantiation:** The loss is instantiated via `KLDivLossConfig::new().init()`.
