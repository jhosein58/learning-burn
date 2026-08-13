# Mean Squared Error (MSE) Loss

This example shows how to calculate the Mean Squared Error (MSE) loss, the default choice for predicting quantitative values like prices, coordinates, or temperatures.

### Key Concepts
- **Outlier Sensitivity:** Squaring the difference between predictions and targets ($[y - \hat{y}]^2$) penalizes large errors disproportionately (e.g., an error of $2$ contributes $4$ times more than an error of $1$). A model trained on MSE will prioritize avoiding massive misses over reducing small, consistent errors.
- **Reductions:** 
  - `Reduction::Mean`: Averages the squared errors over all elements.
  - `Reduction::Sum`: Sums the squared errors without dividing.

### API Mapping & Safety in Burn
- **Type-Safe Enums:** While PyTorch specifies reduction types using strings (e.g., `reduction="mean"`), Burn enforces safety at compile-time with the `Reduction` enum (e.g., `Reduction::Mean`), preventing runtime typos and bugs.
- **Instance-Based API:** Rather than calling a functional module, Burn structures loss calculations using a struct instance. You instantiate the loss helper using `MseLoss::new()` and compute the value via its `.forward(...)` method.
