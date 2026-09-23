use burn::backend::NdArray;
use burn::nn::pool::{AdaptiveAvgPool2dConfig, AvgPool2dConfig, MaxPool2dConfig};
use burn::tensor::Tensor;

type Backend = NdArray;

// How much of the input image one output pixel can "see", layer by layer.
//   rf   grows by (k - 1) * jump
//   jump (distance between neighbouring outputs, in input pixels) *= stride
fn receptive_fields(layers: &[(&str, usize, usize)]) -> Vec<(String, usize)> {
    let (mut rf, mut jump) = (1, 1);
    layers
        .iter()
        .map(|&(name, k, stride)| {
            rf += (k - 1) * jump;
            jump *= stride;
            (name.to_string(), rf)
        })
        .collect()
}

/// Returns (max pool, avg pool, global avg, receptive fields).
fn build() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<(String, usize)>) {
    let device = Default::default();
    // 1..16 laid out as a 4x4 image.
    let x = Tensor::<Backend, 1>::from_floats(
        [
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ],
        &device,
    )
    .reshape([1, 1, 4, 4]);

    let max = MaxPool2dConfig::new([2, 2]).init().forward(x.clone()); // [1,1,2,2]
    let avg = AvgPool2dConfig::new([2, 2]).init().forward(x.clone()); // [1,1,2,2]
    let global = AdaptiveAvgPool2dConfig::new([1, 1]).init().forward(x); // [1,1,1,1]

    // The stack this chapter's CNN uses: conv 3x3, pool 2x2 (stride 2), conv 3x3.
    let rf = receptive_fields(&[("conv 3x3", 3, 1), ("pool 2x2", 2, 2), ("conv 3x3", 3, 1)]);

    (
        max.into_data().to_vec().unwrap(),
        avg.into_data().to_vec().unwrap(),
        global.into_data().to_vec().unwrap(),
        rf,
    )
}

fn main() {
    let (max, avg, global, rf) = build();
    println!("max pool 2x2    = {max:?}");
    println!("avg pool 2x2    = {avg:?}");
    println!("global avg pool = {global:?}");
    for (name, size) in rf {
        println!("after {name}: each output sees {size}x{size} input pixels");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (max, avg, global, rf) = build();
        assert_eq!(max, vec![6.0, 8.0, 14.0, 16.0]);
        assert_eq!(avg, vec![3.5, 5.5, 11.5, 13.5]);
        assert_eq!(global, vec![8.5]);
        let sizes: Vec<usize> = rf.iter().map(|r| r.1).collect();
        assert_eq!(sizes, vec![3, 4, 8]);
    }
}
