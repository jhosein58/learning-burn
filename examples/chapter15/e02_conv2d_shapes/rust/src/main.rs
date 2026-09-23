use burn::backend::NdArray;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2dConfig;
use burn::tensor::Tensor;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;

type Backend = NdArray;

// The rule every conv layer obeys, one spatial axis at a time:
//   out = (n + 2*padding - dilation*(k - 1) - 1) / stride + 1
fn out_len(n: usize, k: usize, padding: usize, stride: usize, dilation: usize) -> usize {
    (n + 2 * padding - dilation * (k - 1) - 1) / stride + 1
}

/// Returns (edge map, [(name, predicted, actual)], weight dims).
fn build() -> (Vec<f32>, Vec<(&'static str, usize, usize)>, [usize; 4]) {
    let device = Default::default();

    // 1. A 5x5 image: dark on the left, bright on the right.
    let image = Tensor::<Backend, 2>::from_floats(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        &device,
    )
    .reshape([1, 1, 5, 5]); // [batch, channels, height, width]

    // A vertical-edge kernel: right column minus left column.
    let kernel = Tensor::<Backend, 2>::from_floats(
        [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
        &device,
    )
    .reshape([1, 1, 3, 3]); // [out_ch, in_ch, kh, kw]

    let edges = conv2d(
        image,
        kernel,
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    );
    println!("edge map shape = {:?}", edges.dims()); // [1, 1, 3, 3]
    let edge_values: Vec<f32> = edges.into_data().to_vec().unwrap();

    // 2. The shape rule, checked against the real nn::Conv2d layer.
    let x = Tensor::<Backend, 4>::zeros([1, 1, 12, 12], &device);
    let cases = [
        ("valid, stride 1", 0, 1, 1),
        ("padding 1", 1, 1, 1),
        ("padding 1, stride 2", 1, 2, 1),
        ("dilation 2", 0, 1, 2),
    ];
    let mut shapes = Vec::new();
    let mut weight_dims = [0; 4];
    for (name, p, s, d) in cases {
        let conv = Conv2dConfig::new([1, 8], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(p, p, p, p))
            .with_stride([s, s])
            .with_dilation([d, d])
            .init::<Backend>(&device);
        weight_dims = conv.weight.dims();
        let got = conv.forward(x.clone()).dims()[2];
        shapes.push((name, out_len(12, 3, p, s, d), got));
    }
    (edge_values, shapes, weight_dims)
}

fn main() {
    let (edges, shapes, weight_dims) = build();
    println!("edge map       = {edges:?}");
    println!("conv weight    = {weight_dims:?}  (out, in, kh, kw)");
    for (name, predicted, actual) in shapes {
        println!("12x12, 3x3, {name:<20} -> rule {predicted:>2}, Conv2d {actual:>2}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (edges, shapes, weight_dims) = build();
        // Same numbers as F.conv2d and nn.Conv2d in python.py.
        assert_eq!(edges, vec![3.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0, 0.0]);
        assert_eq!(weight_dims, [8, 1, 3, 3]);
        let sizes: Vec<usize> = shapes.iter().map(|s| s.2).collect();
        assert_eq!(sizes, vec![10, 12, 6, 8]);
        assert!(shapes.iter().all(|s| s.1 == s.2));
    }
}
