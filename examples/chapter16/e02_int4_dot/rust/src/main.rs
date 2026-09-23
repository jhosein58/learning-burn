// Chapter 16 -- The int4 dot product at the heart of the firmware.
//
// esp32-tinyLLM stores every weight as a 4-bit code: value + 8, so -8..=7
// becomes 0..=15, and two codes share one byte (low nibble first). Its
// hottest loop, dot_int4_int16, multiplies those codes by i16 activations.
// This example is that idea at eight-element scale, with no framework, so
// every number can be checked by hand.

/// Pack weights in -8..=7 into bytes: two nibbles each, stored as value + 8.
fn pack(weights: &[i8]) -> Vec<u8> {
    weights
        .chunks(2)
        .map(|p| ((p[0] + 8) as u8) | (((p[1] + 8) as u8) << 4))
        .collect()
}

/// The obvious way: unpack each nibble, subtract 8, multiply, add.
/// One subtraction per weight.
fn dot_naive(codes: &[u8], acts: &[i16]) -> i32 {
    let mut acc = 0i32;
    for (i, &a) in acts.iter().enumerate() {
        let nibble = if i % 2 == 0 {
            codes[i / 2] & 0xF
        } else {
            codes[i / 2] >> 4
        };
        acc += (nibble as i32 - 8) * a as i32;
    }
    acc
}

/// The firmware's way: multiply the raw nibbles, and subtract 8 * sum(acts)
/// ONCE at the end. sum((n - 8) * a) = sum(n * a) - 8 * sum(a).
/// The activation sum is shared by every row, so it is computed once per token.
fn dot_hoisted(codes: &[u8], acts: &[i16], act_sum: i32) -> i32 {
    let mut acc = 0i32;
    for (p, &b) in codes.iter().enumerate() {
        acc += (b & 0xF) as i32 * acts[2 * p] as i32;
        acc += (b >> 4) as i32 * acts[2 * p + 1] as i32;
    }
    acc - 8 * act_sum
}

/// Returns (packed bytes, naive, hoisted, dequantised float result).
fn build() -> (Vec<u8>, i32, i32, f32) {
    let weights: [i8; 8] = [3, -2, 7, -8, 0, 5, -1, 4];
    let acts: [i16; 8] = [10, -3, 25, 7, -12, 4, 9, -6];
    let codes = pack(&weights);

    let act_sum: i32 = acts.iter().map(|&a| a as i32).sum();
    let naive = dot_naive(&codes, &acts);
    let hoisted = dot_hoisted(&codes, &acts, act_sum);

    // Back to real numbers: one scale for the weight group, one for the activations.
    let (w_scale, a_scale) = (0.05f32, 0.02f32);
    (codes, naive, hoisted, hoisted as f32 * w_scale * a_scale)
}

fn main() {
    let (codes, naive, hoisted, real) = build();
    let hex: Vec<String> = codes.iter().map(|b| format!("0x{b:02X}")).collect();
    println!("8 weights -> {} bytes: {}", codes.len(), hex.join(" "));
    println!("naive dot   = {naive}");
    println!("hoisted dot = {hoisted}");
    println!("as a float  = {real}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_numpy() {
        let (codes, naive, hoisted, real) = build();
        assert_eq!(codes, vec![0x6B, 0x0F, 0xD8, 0xC7]);
        assert_eq!(naive, 142);
        assert_eq!(hoisted, naive); // integer maths: exactly equal, not "close"
        assert!((real - 0.142).abs() < 1e-6);
    }
}
