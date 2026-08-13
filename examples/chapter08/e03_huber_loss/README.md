# Robust Regression: Huber Loss

This example introduces Huber loss (and its special case, Smooth $L_1$), which provides a robust alternative to MSE when dealing with outliers.

### Key Concepts
- **Outlier Mitigation:** While MSE squares errors indiscriminately, Huber loss handles errors dynamically based on a threshold parameter $\delta$:
  $$\text{Huber}(e) = \begin{cases} \frac{1}{2}e^2 & \text{if } |e| \le \delta \\ \delta(|e| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
- **Spliced Behavior:** Inside the transition band ($|e| \le \delta$), Huber behaves quadratically (like MSE) to provide a smooth, easily optimized bowl near the origin. Outside the band ($|e| > \delta$), it scales linearly (like $L_1$ absolute error), ensuring that wild outliers cannot disproportionately dominate the loss gradients.
- **Smooth $L_1$ Relation:** In PyTorch, `smooth_l1_loss` is conceptually identical to Huber loss with the transition threshold $\delta$ locked at $1.0$.

### API Mapping & Implementation in Burn
- **Config Pattern with Delta:** In Burn, the Huber loss is instantiated using a configuration pattern where $\delta$ is passed directly to the constructor: `HuberLossConfig::new(1.0).init()`.
- **Target Cases:** Huber loss is highly recommended for regression tasks with inherently noisy datasets, such as financial time-series or sensor data containing uncleaned anomalies.
