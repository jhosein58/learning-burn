# Chapter 18 weights

Written by Example 18.2 (`cargo run --release -p c18e2`), which retrains them from the same fixed starting weights and data stream, deterministically:

- `flow_unet.bpk` — the trained U-Net, burnpack (Examples 18.3–18.5 load it).
- `flow_unet.safetensors` — the same weights with PyTorch names and layout (burn-store, `BurnToPyTorchAdapter`), for the Python sides.
- `flow_unet_loss.log` — the Learner's per-step training loss log.

Example 18.4 adds `flow_unet_reflow.bpk`, the model after one round of reflow (not committed; rerun Example 18.4 to make it).
