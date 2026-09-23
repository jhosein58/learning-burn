# Chapter 16 — Firmware in Rust

The firmware itself lives in its own repository:
**[esp32-tinyLLM](https://github.com/Bellman281/esp32-tinyLLM)** (the chapter quotes commit `f497f50`).
It needs an ESP32-S3-DevKitC **N16R8** (16 MB flash, 8 MB PSRAM) for the on-chip part;
its host test suite (`cd engine/llm-host && cargo test --release`) runs on any laptop.

The two examples here run on a laptop and need nothing else:

| Example | What it shows |
|---|---|
| [`e01_will_it_fit`](e01_will_it_fit/) | the memory budget of a 28.9M-parameter model, to the byte |
| [`e02_int4_dot`](e02_int4_dot/) | packed int4 weights × i16 activations, with the offset hoisted |
