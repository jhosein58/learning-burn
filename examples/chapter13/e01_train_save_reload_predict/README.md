# Training, Saving, and Inference Round Trip (Example 13.1)

A complete end-to-end pipeline demonstrating how to train a model, serialize its weights to disk, and reload them for safe, lightweight inference in **Burn** and **PyTorch**.

## The Core Concept
Modern deep learning workflows separate training from deployment. In Burn, this division is enforced compile-time by the Rust type system:
- **Training Mode:** Executed under the `Autodiff<Backend>` type wrapper to build and track the computational graph for gradient updates.
- **Inference Mode:** Uses the plain `Backend` type (without `Autodiff`). 

This architectural division eliminates the need for a runtime `.eval()` flag.

## Key Takeaways
- **No More `.eval()` Bugs:** In PyTorch, forgetting `.eval()` keeps dropout and batch normalization active during inference, leading to silent bugs. In Burn, the inference model is compiled without the Autodiff wrapper, meaning a backward pass or training-only behavior is physically impossible.
- **Ownership on Save:** The `.save_file()` function in Burn consumes the model by value. To continue using or testing a model after saving, you must clone it before serialization.
- **Recorders:** Burn uses a structured `Recorder` (e.g., `CompactRecorder`) to serialize and map the model's weight state back onto a freshly initialized model structure.
