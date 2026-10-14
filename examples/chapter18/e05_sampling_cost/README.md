# Example 18.5 — What a sample costs

Times the unguided (w = 1, n images per step) and guided (2n images per step) Euler samplers for 1 to 64 steps, 64 images per call, and fits a line: fixed cost plus cost per step. The device is a value: built with `--features gpu`, the same function also runs on `Device::wgpu`.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference (timed with PyTorch on the CPU) + `pytest` test. |

## Run & test

```bash
cargo run -- -c 18 -e 5 rust      # from the repo root
cargo run --release -p c18e5 --features gpu   # adds a GPU row (Metal, Vulkan or DX12)
cargo run -- -c 18 -e 5 python
cargo test -p c18e5 --release     # the Rust test
pytest examples/chapter18/e05_sampling_cost/python/main.py
```

## Parity

Timings are per machine. The test checks the samples instead: four 2-step samples match PyTorch to 1e-3, and guidance with w = 1 equals the unguided sampler.
