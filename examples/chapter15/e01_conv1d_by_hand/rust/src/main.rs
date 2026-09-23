use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;

type Backend = NdArray;

// A convolution is a small kernel sliding along the signal. At every position
// it multiplies the overlapping values and adds them up. That's all it is.
fn by_hand(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
    let k = kernel.len();
    (0..=signal.len() - k)
        .map(|i| (0..k).map(|j| signal[i + j] * kernel[j]).sum())
        .collect()
}

/// Returns (by-hand result, Burn's conv1d result).
fn build() -> (Vec<f32>, Vec<f32>) {
    let device = Default::default();
    let signal = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0];
    let kernel = [1.0, 0.0, -1.0]; // "left minus right": a slope detector

    let manual = by_hand(&signal, &kernel);

    // Burn wants [batch, channels, length] for the input and
    // [out_channels, in_channels, kernel_len] for the weight.
    let x = Tensor::<Backend, 1>::from_floats(signal, &device).reshape([1, 1, 6]);
    let w = Tensor::<Backend, 1>::from_floats(kernel, &device).reshape([1, 1, 3]);
    let y = conv1d(x, w, None, ConvOptions::new([1], [0], [1], 1));
    println!("burn shape = {:?}", y.dims()); // [1, 1, 4]

    (manual, y.into_data().to_vec().unwrap())
}

fn main() {
    let (manual, burn) = build();
    println!("by hand    = {manual:?}");
    println!("burn conv1d= {burn:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (manual, burn) = build();
        // PyTorch's F.conv1d gives the same four numbers.
        assert_eq!(manual, vec![-3.0, -5.0, -7.0, -9.0]);
        assert_eq!(burn, manual);
    }
}
