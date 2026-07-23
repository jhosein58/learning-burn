# Backend-Agnostic Inference (Example 14.2)

An architectural demonstration showing how **Burn**'s type system decouples model logic from execution hardware, allowing the exact same code to run seamlessly on CPU, GPU, or embedded microcontrollers.

## The Core Concept
In PyTorch, device allocation (such as `.to("cuda")` or `.to("cpu")`) is handled via runtime values. This makes it easy to write code that compiles perfectly but crashes hours into a training job with a `RuntimeError` due to a device mismatch.

In Burn, the target hardware is represented as a compile-time **Type Parameter** (`B: Backend`). 

## Key Architectural Advantages
- **Compile-Time Device Safety:** Because the backend is encoded directly into the Tensor's Rust type, you physically cannot mix tensors from different hardware devices. Mismatches are caught immediately by the compiler.
- **Write Once, Run Anywhere:** Your model code, forward passes, and inference loops do not care about the hardware. You write the logic generically and select the concrete backend (e.g., CPU `NdArray`, GPU `Wgpu`, or an embedded `no_std` backend) at the call site.
- **No-Risk Scaling:** This trait-based abstraction allows developers to build, test, and debug models on a local CPU laptop and deploy the exact same compiled binary onto a high-performance GPU cluster or a resource-constrained embedded chip.
