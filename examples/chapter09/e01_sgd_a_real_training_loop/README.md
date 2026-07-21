# SGD: A Real Training Loop

This example demonstrates how to build and execute a complete training loop in Burn using Stochastic Gradient Descent (SGD) to solve a simple linear regression task ($y = 2x + 1$).

### Key Concepts & Differences
- **Functional State Updates:** PyTorch optimizers hold mutable references to model parameters and mutate them in-place behind the scenes. To satisfy Rust's strict ownership and borrowing rules, Burn avoids shared mutable state:
  - `GradientsParams::from_grads(grads, &model)` maps raw backpropagated gradients directly to the model's parameters.
  - `optim.step(lr, model, grads)` consumes the old model and returns an entirely new, updated model instance.
- **No `zero_grad()` Bug:** In PyTorch, gradients accumulate in-place, making a forgotten `.zero_grad()` call a frequent source of silent bugs. In Burn, `loss.backward()` returns a brand-new gradients container on every pass. There is no accumulated state to clear, making this category of bug impossible to write.
- **Model Definition:** A custom model is declared by deriving the `Module` trait on a struct wrapping Burn's basic building blocks (like `Linear`).

### The Four Beats of a Burn Training Loop
1. **Forward:** `model.forward(x.clone())` computes predictions (cloning inputs to preserve them for future iterations).
2. **Loss:** `MseLoss::new().forward(pred, y.clone(), Reduction::Mean)` evaluates prediction error.
3. **Backward:** `loss.backward()` computes gradients and returns them in a fresh container.
4. **Step:** `model = optim.step(lr, model, grads)` consumes the model and gradients to yield the updated parameters.
