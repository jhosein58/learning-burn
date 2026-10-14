// Chapter 23 -- Frames, hops and the price of forgetting.
//
// A keyword spotter on a microcontroller sees audio as a stream of frames: a
// 25 ms frame every 10 ms. Each frame brings one new output and a deadline.
// This example prices one model three ways -- recompute the whole window every
// hop, recompute only the receptive field, or keep state and stream -- with no
// framework, just the formulas of Section 23.1.

const SAMPLE_RATE: usize = 16_000; // Hz
const FRAME: usize = 400; //          25 ms of samples
const HOP: usize = 160; //            10 ms of samples
const CLIP: usize = 16_000; //        one second

const FEATURES: usize = 40; // features per frame (Chapter 24 computes them)
const CHANNELS: usize = 64; // channels of every conv layer
const K: usize = 3; //         kernel size
const DILATIONS: [usize; 4] = [1, 2, 4, 8];

const CLOCK_HZ: usize = 240_000_000; // ESP32-S3 core clock (Chapter 22)

fn ms(samples: usize) -> f64 {
    samples as f64 * 1000.0 / SAMPLE_RATE as f64
}

/// Number of whole frames in a clip: 1 + floor((n - frame) / hop).
fn n_frames(n: usize) -> usize {
    1 + (n - FRAME) / HOP
}

/// (c_in, c_out, dilation) for each layer.
fn layers() -> Vec<(usize, usize, usize)> {
    DILATIONS
        .iter()
        .enumerate()
        .map(|(l, &d)| (if l == 0 { FEATURES } else { CHANNELS }, CHANNELS, d))
        .collect()
}

/// Everything the example prints, so the test can check it.
#[allow(dead_code)] // the fields are read by the test
struct Budget {
    frames: usize,
    receptive: usize,
    macs_stream: usize,
    macs_cone: usize,
    macs_window: usize,
    state_floats: usize,
    window_floats: usize,
    params: usize,
}

fn build() -> Budget {
    let frames = n_frames(CLIP);
    let layers = layers();

    // Receptive field: 1 + sum over layers of d (k - 1) frames.
    let receptive = 1 + layers.iter().map(|&(_, _, d)| d * (K - 1)).sum::<usize>();

    // One output frame of one layer costs c_out * c_in * k multiply-adds.
    let macs: Vec<usize> = layers.iter().map(|&(ci, co, _)| co * ci * K).collect();
    let macs_stream: usize = macs.iter().sum();

    // Recompute only the cone: layer l must produce n_l outputs, where the last
    // layer produces 1 and each layer below needs d (k - 1) more.
    let mut need = 1;
    let mut macs_cone = 0;
    for (l, &(_, _, d)) in layers.iter().enumerate().rev() {
        macs_cone += need * macs[l];
        need += d * (K - 1);
    }
    assert_eq!(need, receptive); // the cone's base is the receptive field

    // Recompute the whole window (the clip) every hop.
    let macs_window = frames * macs_stream;

    // State to keep between frames: the last d (k - 1) inputs of every layer.
    let state_floats: usize = layers.iter().map(|&(ci, _, d)| d * (K - 1) * ci).sum();
    // Without state: the input window plus two activation maps alive at once.
    let window_floats = frames * FEATURES + 2 * frames * CHANNELS;

    let params: usize = layers.iter().map(|&(ci, co, _)| co * ci * K + co).sum();

    println!(
        "audio     {SAMPLE_RATE} Hz, frame {} ms = {FRAME} samples, hop {} ms = {HOP} samples",
        ms(FRAME),
        ms(HOP)
    );
    println!("clip 1 s  -> {frames} frames");
    println!(
        "model     {FEATURES} features -> {} causal convs, {CHANNELS} channels, k = {K}, dilations {DILATIONS:?}",
        DILATIONS.len()
    );
    println!(
        "receptive field = {receptive} frames = {} ms of audio",
        ms((receptive - 1) * HOP + FRAME)
    );
    println!(
        "look-ahead with 'same' padding = {} frames = {} ms",
        (receptive - 1) / 2,
        ms((receptive - 1) / 2 * HOP)
    );
    println!("layer  c_in  c_out  d  MACs/frame  state floats");
    for (l, &(ci, co, d)) in layers.iter().enumerate() {
        println!(
            "{:>5} {ci:>5} {co:>6} {d:>2} {:>11} {:>13}",
            l + 1,
            macs[l],
            d * (K - 1) * ci
        );
    }
    let hops_per_s = SAMPLE_RATE / HOP;
    println!("MACs per hop, streaming        = {macs_stream:>9}");
    println!(
        "MACs per hop, recompute cone   = {macs_cone:>9}  ({:.1}x)",
        macs_cone as f64 / macs_stream as f64
    );
    println!(
        "MACs per hop, recompute window = {macs_window:>9}  ({}x, {frames} frames)",
        macs_window / macs_stream
    );
    println!(
        "per second ({hops_per_s} hops): streaming {:.2} M, recompute {:.1} M",
        (macs_stream * hops_per_s) as f64 / 1e6,
        (macs_window * hops_per_s) as f64 / 1e6
    );
    println!(
        "cycles per hop at {} MHz = {}",
        CLOCK_HZ / 1_000_000,
        CLOCK_HZ / hops_per_s
    );
    println!(
        "state, streaming     = {state_floats} floats = {} bytes",
        state_floats * 4
    );
    println!(
        "window, recompute    = {window_floats} floats = {} bytes",
        window_floats * 4
    );
    println!(
        "weights              = {params} params = {} bytes f32",
        params * 4
    );

    Budget {
        frames,
        receptive,
        macs_stream,
        macs_cone,
        macs_window,
        state_floats,
        window_floats,
        params,
    }
}

fn main() {
    build();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // main.py computes the same budget; these are its numbers.
        let b = build();
        assert_eq!(b.frames, 98);
        assert_eq!(b.receptive, 31);
        assert_eq!(b.macs_stream, 44_544);
        assert_eq!(b.macs_cone, 751_104);
        assert_eq!(b.macs_window, 4_365_312);
        assert_eq!(b.state_floats, 1_872);
        assert_eq!(b.window_floats, 16_464);
        assert_eq!(b.params, 44_800);
    }
}
