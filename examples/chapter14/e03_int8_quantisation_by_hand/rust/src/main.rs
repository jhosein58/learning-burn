// Chapter 14 — Quantization: shrinking weights from f32 to int8 (~4x smaller).
//
// Symmetric int8 quantization, the standard scheme:
//      scale = max(|w|) / 127
//      q     = round(w / scale)-> stored as i8, one byte each
//      w'    = q * scale-> dequantize at inference time
// You keep a single f32 `scale` per tensor and store every weight in one byte.
// Reconstruction is approximate, but usually accurate enough for inference — and
// on a constrained device the 4x memory saving is what makes the model fit.
//
// This is pure arithmetic (no framework) so the mechanism is fully visible.

fn main() {
    let weights: [f32; 5] = [0.5, -1.2, 0.03, 0.9, -0.4];

    // 1. scale from the largest magnitude weight.
    let max_abs = weights.iter().fold(0.0f32, |m, &w| m.max(w.abs())); // 1.2
    let scale = max_abs / 127.0;

    // 2. quantize to int8 codes.
    let q: Vec<i8> = weights.iter().map(|&w| (w / scale).round() as i8).collect();

    // 3. dequantize back to approximate floats.
    let dequant: Vec<f32> = q.iter().map(|&c| c as f32 * scale).collect();
    let max_err = weights
        .iter()
        .zip(&dequant)
        .map(|(&w, &d)| (w - d).abs())
        .fold(0.0f32, f32::max);

    println!("original    = {weights:?}");
    println!("scale       = {scale:.6}");
    println!("int8 codes  = {q:?}"); // [53, -127, 3, 95, -42]
    println!("dequantized = {dequant:?}");
    println!("max error   = {max_err:.5}"); // ~0.00315

    let f32_bytes = weights.len() * 4;
    let int8_bytes = q.len() + 4; // codes + one f32 scale

    println!("size: f32 {f32_bytes} bytes -> int8 {int8_bytes} bytes (~4x smaller)");
    // Burn also ships built-in quantization (Module::quantize_weights) for real
    // models; this example is the arithmetic it performs under the hood.
}
