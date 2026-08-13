# Int8 Quantization (Example 14.3)

A fundamental demonstration of symmetric int8 quantization, showcasing how to shrink neural network weights by approximately 4x with negligible loss in model accuracy.

## The Core Concept
Standard `f32` weights require four bytes each. In resource-constrained systems, this memory footprint is often too large. Quantization maps these continuous 32-bit floats into discrete 8-bit signed integers (`i8`), reducing storage requirements to just one byte per weight.

## How It Works
1. **Find the Scale:** Calculate a single scaling factor (`scale = max(|weights|) / 127`) to map the maximum absolute value in the tensor to the limit of an 8-bit integer.
2. **Quantize:** Divide weights by the scale and round them to the nearest integer, producing values between `-127` and `127` that fit perfectly into 1 byte.
3. **Dequantize:** During inference, multiply the stored `i8` codes back by the `scale` to reconstruct approximate floats.

## Key Advantages
- **4x Memory Savings:** Fits larger models onto microcontrollers and embedded devices with limited SRAM/Flash.
- **Symmetric Design:** Using `127` instead of `128` as the divisor ensures perfect symmetry, guaranteeing that a weight of exactly `0.0` maps perfectly back to `0.0`.
- **Negligible Error:** Reconstruction error is typically less than 0.3%, allowing inference to run with virtually identical performance.
