// Chapter 16 -- "Will it fit?", for a real model.
//
// Chapter 14 asked the question for a 2,410-parameter MLP. Here is the same
// arithmetic for the language model this chapter puts on an ESP32-S3: count
// every tensor from the config, price it in bytes, and hold it up against the
// board's three kinds of memory. No framework, just the numbers.

// The model's config, as printed in its own header (esp32-tinyLLM, model.bin).
const V: usize = 32768; // vocabulary rows
const D: usize = 96; //    model width
const L: usize = 6; //     layers
const F: usize = 66; //    feed-forward width
const P: usize = 128; //   per-layer-embedding width
const GROUP: usize = 128; // int4 weights share one fp16 scale per 128 values

// The board: ESP32-S3-DevKitC N16R8.
const SRAM: usize = 512 * 1024;
const PSRAM: usize = 8 * 1024 * 1024;
const FLASH: usize = 16 * 1024 * 1024;

/// A weight matrix: (rows, cols). A norm vector: (n, 0).
fn tensors() -> Vec<(&'static str, usize, usize)> {
    let mut t = vec![
        ("token embedding", V, D),
        ("PLE model proj", L * P, D),
        ("PLE proj norm", P, 0),
        ("PLE table", V, L * P),
    ];
    for _ in 0..L {
        t.extend([
            ("attn norm", D, 0),
            ("qkv", 3 * D, D),
            ("attn proj", D, D),
            ("ffn norm", D, 0),
            ("gate", F, D),
            ("up", F, D),
            ("down", D, F),
            ("ple gate", P, D),
            ("ple proj", D, P),
            ("ple norm", D, 0),
        ]);
    }
    t.push(("out norm", D, 0));
    t
}

fn params(rows: usize, cols: usize) -> usize {
    if cols == 0 { rows } else { rows * cols }
}

// Bytes on disk: int4 codes (two per byte) + one fp16 scale per group + a
// 4-byte group field per matrix; norms stay f32. Plus a 40-byte header.
fn int4_bytes(rows: usize, cols: usize) -> usize {
    if cols == 0 {
        return rows * 4;
    }
    4 + rows * cols.div_ceil(2) + rows * cols.div_ceil(GROUP) * 2
}

/// Returns (total params, PLE-table params, f32 bytes, int4 file bytes).
fn build() -> (usize, usize, usize, usize) {
    let t = tensors();
    let total: usize = t.iter().map(|&(_, r, c)| params(r, c)).sum();
    let ple = params(V, L * P);
    let file = 40 + t.iter().map(|&(_, r, c)| int4_bytes(r, c)).sum::<usize>();
    (total, ple, total * 4, file)
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1e6
}

fn main() {
    let (total, ple, f32_bytes, file) = build();
    println!("parameters      = {total}");
    println!(
        "  in PLE table  = {ple} ({:.1}%)",
        100.0 * ple as f64 / total as f64
    );
    println!("f32 weights     = {:.1} MB", mb(f32_bytes));
    println!("int4 model.bin  = {file} bytes ({:.2} MB)", mb(file));
    println!();
    for (name, size) in [("SRAM", SRAM), ("PSRAM", PSRAM), ("flash", FLASH)] {
        let verdict = |b: usize| if b <= size { "fits" } else { "does NOT fit" };
        println!(
            "{name:<5} {:>5.2} MB: f32 {:<12} int4 {}",
            mb(size),
            verdict(f32_bytes),
            verdict(file)
        );
    }
    // The part the CPU must stream on EVERY token: the output head, which is
    // the token embedding reused (tied), capped to the 25,353 real vocab entries.
    let head_codes = 25_353 * D.div_ceil(2);
    println!();
    println!("head codes streamed per token = {head_codes} bytes");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_and_the_real_file() {
        let (total, ple, f32_bytes, file) = build();
        assert_eq!(total, 28_869_920); // "28.9M parameters"
        assert_eq!(ple, 25_165_824); //   "~25M live in the PLE table"
        assert_eq!(f32_bytes, 115_479_680);
        assert_eq!(file, 14_912_332); //  ls -l model/tinystories-v32768/model.bin
        assert_eq!(25_353 * D.div_ceil(2), 1_216_944); // the firmware's "1.22 MB of head weights"
    }
}
