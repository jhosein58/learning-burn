use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor};

type Backend = NdArray;

fn uniform(n: usize) -> Tensor<Backend, 1> {
    let device = Default::default();
    Tensor::<Backend, 1>::random([n], Distribution::Default, &device)
}

fn normal(n: usize) -> Tensor<Backend, 1> {
    let device = Default::default();
    Tensor::<Backend, 1>::random([n], Distribution::Normal(0.0, 1.0), &device)
}

fn main() {
    let device = Default::default();
    // demo: a small 2x3 draw from each distribution
    let u = Tensor::<Backend, 2>::random([2, 3], Distribution::Default, &device);
    let n = Tensor::<Backend, 2>::random([2, 3], Distribution::Normal(0.0, 1.0), &device);
    println!("uniform = {}", u);
    println!("normal = {}", n);
}

// Random draws will NEVER match PyTorch element-for-element (different RNGs).
// Parity is enforced on the DISTRIBUTION instead: both frameworks must obey the
// same contract, checked over a large sample with a loose (many-sigma) tolerance.
#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 100_000;

    #[test]
    fn uniform_contract() {
        let t = uniform(N);
        let lo = t.clone().min().into_scalar();
        let hi = t.clone().max().into_scalar();
        let mean = t.mean().into_scalar();

        assert!(lo >= 0.0, "uniform min {lo} < 0");
        assert!(hi < 1.0, "uniform max {hi} >= 1");
        assert!((mean - 0.5).abs() < 0.05, "uniform mean {mean} far from 0.5");
    }

    #[test]
    fn normal_contract() {
        let t = normal(N);
        let mean = t.clone().mean().into_scalar();
        // std = sqrt(mean((x - mean)^2)); Burn has no direct std().
        let std = t.sub_scalar(mean).powf_scalar(2.0).mean().sqrt().into_scalar();

        assert!(mean.abs() < 0.05, "normal mean {mean} far from 0");
        assert!((std - 1.0).abs() < 0.05, "normal std {std} far from 1");
    }
}
