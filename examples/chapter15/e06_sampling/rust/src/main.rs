use burn::tensor::activation::softmax;
use burn::tensor::{Device, Tensor};

// Decoding: four ways to turn one logits vector into the next token.
// Each strategy is a function from logits to a probability vector; drawing a
// token is then the inverse CDF of that vector at a uniform number u.
const Z: [f32; 5] = [2.0, 1.0, 0.5, 0.0, -1.0];
const TOKENS: [&str; 5] = ["a", "b", "c", "d", "e"];
const DRAWS: usize = 10_000;

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF
    }
    fn uniform(&mut self) -> f32 {
        self.next() as f32 / 32768.0
    }
}

// Greedy: the most likely token, every time. No randomness at all.
fn greedy(z: Tensor<1>) -> usize {
    z.argmax(0).into_scalar::<i32>() as usize
}

// Temperature: divide the logits by T before the softmax.
fn with_temperature(z: Tensor<1>, temp: f64) -> Tensor<1> {
    softmax(z.div_scalar(temp), 0)
}

// Top-k: keep the k largest logits, push the rest to -1e9, then softmax.
fn top_k(z: Tensor<1>, k: usize, temp: f64) -> Tensor<1> {
    let kth: f32 = z.clone().sort_descending(0).slice([k - 1..k]).into_scalar();
    let z = z.clone().mask_fill(z.lower_elem(kth), -1.0e9);
    with_temperature(z, temp)
}

// Top-p (nucleus): keep the most likely tokens until their total reaches p,
// i.e. every token whose predecessors in sorted order sum to less than p.
fn top_p(z: Tensor<1>, p: f64, temp: f64) -> Tensor<1> {
    let probs = with_temperature(z, temp);
    let sorted = probs.clone().sort_descending(0);
    // the probability mass ahead of each token in sorted order
    let before = sorted.clone().cumsum(0) - sorted.clone();
    let kept: i32 = before.lower_elem(p).int().sum().into_scalar();
    let kept = kept as usize;
    let smallest: f32 = sorted.slice([kept - 1..kept]).into_scalar();
    let probs = probs.clone().mask_fill(probs.lower_elem(smallest), 0.0);
    probs.clone() / probs.sum() // renormalise over the nucleus
}

// Inverse CDF: the first token whose cumulative probability exceeds u.
fn sample(probs: &[f32], u: f32) -> usize {
    let mut cdf = 0.0;
    for (i, &q) in probs.iter().enumerate() {
        cdf += q;
        if u < cdf {
            return i;
        }
    }
    probs.len() - 1 // u fell in the rounding gap above the last cdf value
}

fn entropy(p: &[f32]) -> f32 {
    p.iter().filter(|&&q| q > 0.0).map(|&q| -q * q.ln()).sum()
}

struct Strategy {
    name: &'static str,
    probs: Vec<f32>,
    draws: Vec<usize>,
    counts: [usize; 5],
}

fn build() -> (usize, Vec<f32>, Vec<Strategy>) {
    let dev = Device::flex();
    let z = || Tensor::<1>::from_floats(Z, &dev);
    let choices: [(&str, Tensor<1>); 5] = [
        ("T = 1", with_temperature(z(), 1.0)),
        ("T = 0.5", with_temperature(z(), 0.5)),
        ("T = 2", with_temperature(z(), 2.0)),
        ("top-k 2", top_k(z(), 2, 1.0)),
        ("top-p 0.8", top_p(z(), 0.8, 1.0)),
    ];
    let mut first_u = Vec::new();
    let strategies = choices
        .into_iter()
        .map(|(name, p)| {
            let probs: Vec<f32> = p.into_data().try_to_vec().unwrap();
            let mut rng = Lcg(5); // the same uniforms for every strategy
            let us: Vec<f32> = (0..DRAWS).map(|_| rng.uniform()).collect();
            let all: Vec<usize> = us.iter().map(|&u| sample(&probs, u)).collect();
            let mut counts = [0; 5];
            all.iter().for_each(|&i| counts[i] += 1);
            first_u = us[..10].to_vec();
            Strategy {
                name,
                probs,
                draws: all[..10].to_vec(),
                counts,
            }
        })
        .collect();
    (greedy(z()), first_u, strategies)
}

fn main() {
    let (g, us, strategies) = build();
    println!("logits z = {Z:?}  tokens {TOKENS:?}");
    println!("greedy -> {}", TOKENS[g]);
    let us: Vec<String> = us.iter().map(|u| format!("{u:.3}")).collect();
    println!("first 10 uniforms u = {}", us.join(" "));
    println!("strategy     p(a)   p(b)   p(c)   p(d)   p(e)   entropy");
    for s in &strategies {
        let p: Vec<String> = s.probs.iter().map(|p| format!("{p:.4}")).collect();
        println!("{:<10}  {}  {:.3}", s.name, p.join(" "), entropy(&s.probs));
    }
    println!("strategy    first 10 draws        counts in {DRAWS} draws");
    for s in &strategies {
        let d: Vec<&str> = s.draws.iter().map(|&i| TOKENS[i]).collect();
        println!("{:<10}  {}   {:?}", s.name, d.join(" "), s.counts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (g, _, strategies) = build();
        assert_eq!(g, 0);
        // From python/main.py: the same functions in torch, the same uniforms.
        let torch: [[f32; 5]; 5] = [
            [0.56302124, 0.20712395, 0.12562703, 0.07619664, 0.02803118],
            [0.82924461, 0.11222605, 0.04128566, 0.01518815, 0.00205549],
            [0.37454489, 0.22717296, 0.17692247, 0.13778737, 0.08357226],
            [0.7310586, 0.26894143, 0.0, 0.0, 0.0],
            [0.62853169, 0.23122391, 0.14024439, 0.0, 0.0],
        ];
        let torch_counts = [
            [5583, 2071, 1260, 781, 305],
            [8258, 1132, 430, 156, 24],
            [3701, 2281, 1765, 1376, 877],
            [7267, 2733, 0, 0, 0],
            [6220, 2346, 1434, 0, 0],
        ];
        for ((s, p), c) in strategies.iter().zip(torch).zip(torch_counts) {
            for (a, b) in s.probs.iter().zip(p) {
                assert!((a - b).abs() < 1e-6, "{}: {a} vs {b}", s.name);
            }
            assert_eq!(s.counts, c, "{}", s.name);
        }
    }
}
