use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{Device, Int, Tensor, TensorData};

// Chapter 16 -- A GPT as a #[derive(Module)]: token and position embeddings, a stack of
// pre-norm transformer blocks with a causal mask, a final LayerNorm and a linear head.

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/shakespeare.txt");

// --- the model, built from its config ---
#[derive(Config, Debug)]
struct GptConfig {
    vocab_size: usize,
    #[config(default = 128)]
    block_size: usize, // the longest context the model can read: T
    #[config(default = 64)]
    d_model: usize,
    #[config(default = 4)]
    n_heads: usize,
    #[config(default = 2)]
    n_layers: usize,
    #[config(default = 0.0)]
    dropout: f64,
}

impl GptConfig {
    fn init(&self, device: &Device) -> Gpt {
        let d = self.d_model;
        Gpt {
            tok: EmbeddingConfig::new(self.vocab_size, d).init(device),
            pos: EmbeddingConfig::new(self.block_size, d).init(device),
            blocks: TransformerEncoderConfig::new(d, 4 * d, self.n_heads, self.n_layers)
                .with_norm_first(true) // pre-norm, as in GPT-2
                .with_dropout(self.dropout)
                .init(device),
            ln_f: LayerNormConfig::new(d).init(device),
            head: LinearConfig::new(d, self.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct Gpt {
    tok: Embedding,             // [V, d]: one row per token id
    pos: Embedding,             // [T, d]: one row per position
    blocks: TransformerEncoder, // L pre-norm blocks
    ln_f: LayerNorm,            // the final norm a pre-norm stack needs
    head: Linear,               // d -> V logits, no bias
}

impl Gpt {
    fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let [n, t] = tokens.dims();
        let device = tokens.device();
        let positions = Tensor::<1, Int>::arange(0..t as i64, &device).reshape([1, t]);
        // [N, T, d] + [1, T, d]: the positions broadcast over the batch
        let x = self.tok.forward(tokens) + self.pos.forward(positions);
        // [N, T, T], true = hidden: query i may not see key j > i
        let causal = generate_autoregressive_mask(n, t, &device);
        let x = self
            .blocks
            .forward(TransformerEncoderInput::new(x).mask_attn(causal));
        self.head.forward(self.ln_f.forward(x)) // [N, T, V]
    }
}

// --- fixed weights from an LCG, in the same order as main.py ---
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    // n values, uniform on [-bound, bound): PyTorch's default range for a Linear layer
    fn uniform(&mut self, n: usize, bound: f64) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() as f64 / 32768.0 * 2.0 - 1.0) * bound) as f32)
            .collect()
    }
}

fn param<const D: usize>(v: Vec<f32>, shape: [usize; D], device: &Device) -> Param<Tensor<D>> {
    Param::from_tensor(Tensor::from_data(TensorData::new(v, shape), device))
}

impl Lcg {
    // Small embeddings: every block starts with a LayerNorm, which rescales them anyway.
    fn embedding(&mut self, e: &mut Embedding, rows: usize, d: usize, device: &Device) {
        e.weight = param(self.uniform(rows * d, 0.1), [rows, d], device);
    }
    // Drawn in PyTorch's [out, in] layout, handed to Burn as [in, out].
    fn linear(&mut self, l: &mut Linear, d_in: usize, d_out: usize, device: &Device) {
        let bound = 1.0 / (d_in as f64).sqrt();
        let w = Tensor::<2>::from_data(
            TensorData::new(self.uniform(d_out * d_in, bound), [d_out, d_in]),
            device,
        );
        l.weight = Param::from_tensor(w.transpose());
        if l.bias.is_some() {
            l.bias = Some(param(self.uniform(d_out, bound), [d_out], device));
        }
    }
    fn norm(&mut self, ln: &mut LayerNorm, d: usize, device: &Device) {
        let g: Vec<f32> = self.uniform(d, 0.1).iter().map(|v| 1.0 + v).collect();
        ln.gamma = param(g, [d], device);
        ln.beta = Some(param(self.uniform(d, 0.1), [d], device));
    }
}

impl Gpt {
    fn with_fixed_weights(mut self, seed: u32, device: &Device) -> Self {
        let [v, d] = self.tok.weight.dims();
        let t = self.pos.weight.dims()[0];
        let mut f = Lcg(seed);
        f.embedding(&mut self.tok, v, d, device);
        f.embedding(&mut self.pos, t, d, device);
        for layer in self.blocks.layers.iter_mut() {
            f.norm(&mut layer.norm_2, d, device); // with norm_first, norm_2 comes BEFORE attention
            f.linear(&mut layer.mha.query, d, d, device);
            f.linear(&mut layer.mha.key, d, d, device);
            f.linear(&mut layer.mha.value, d, d, device);
            f.linear(&mut layer.mha.output, d, d, device);
            f.norm(&mut layer.norm_1, d, device); // ...and norm_1 before the feed-forward
            f.linear(&mut layer.pwff.linear_inner, d, 4 * d, device);
            f.linear(&mut layer.pwff.linear_outer, 4 * d, d, device);
        }
        f.norm(&mut self.ln_f, d, device);
        f.linear(&mut self.head, d, v, device);
        self
    }
}

// --- the input: the first two training windows of Example 16.1 ---
fn windows(device: &Device) -> (Tensor<2, Int>, Tensor<2, Int>, usize) {
    let text = std::fs::read_to_string(DATA).expect("read shakespeare.txt");
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort();
    chars.dedup();
    let ids: Vec<i64> = text
        .chars()
        .map(|c| chars.binary_search(&c).unwrap() as i64)
        .collect();
    let t = 128;
    let rows = |offset: usize| -> Vec<i64> {
        [0, t]
            .iter()
            .flat_map(|&s| ids[s + offset..s + offset + t].to_vec())
            .collect()
    };
    let inputs = Tensor::from_data(TensorData::new(rows(0), [2, t]), device);
    let targets = Tensor::from_data(TensorData::new(rows(1), [2, t]), device);
    (inputs, targets, chars.len())
}

struct Report {
    counts: Vec<(&'static str, usize)>,
    num_params: usize,
    logits_dims: [usize; 3],
    first: Vec<f32>,
    last: Vec<f32>,
    loss: f32,
    causal: (f32, f32),
}

fn build() -> Report {
    let device = Device::flex();
    let (inputs, targets, vocab) = windows(&device);
    let config = GptConfig::new(vocab);
    let model = config.init(&device).with_fixed_weights(42, &device);

    let counts = vec![
        ("token embedding", model.tok.num_params()),
        ("position embedding", model.pos.num_params()),
        ("transformer blocks", model.blocks.num_params()),
        ("final LayerNorm", model.ln_f.num_params()),
        ("head", model.head.num_params()),
    ];

    let logits = model.forward(inputs.clone()); // [2, 128, 62]
    let [n, t, v] = logits.dims();
    let row = |b: usize, p: usize| -> Vec<f32> {
        logits
            .clone()
            .slice([b..b + 1, p..p + 1, 0..4])
            .into_data()
            .try_to_vec()
            .unwrap()
    };
    let (first, last) = (row(0, 0), row(1, t - 1));
    let loss = CrossEntropyLossConfig::new()
        .init(&device)
        .forward(logits.clone().reshape([n * t, v]), targets.reshape([n * t]))
        .into_scalar::<f32>();

    // Causality: change the LAST input token of every row and see which logits move.
    let last_col = inputs.clone().slice([0..n, t - 1..t]);
    let changed = inputs.slice_assign(
        [0..n, t - 1..t],
        last_col.add_scalar(1).remainder_scalar(v as i64),
    );
    let moved = (model.forward(changed) - logits).abs();
    let before = moved
        .clone()
        .slice([0..n, 0..t - 1, 0..v])
        .max()
        .into_scalar::<f32>();
    let at_last = moved
        .slice([0..n, t - 1..t, 0..v])
        .max()
        .into_scalar::<f32>();

    Report {
        counts,
        num_params: model.num_params(),
        logits_dims: [n, t, v],
        first,
        last,
        loss,
        causal: (before, at_last),
    }
}

fn main() {
    let r = build();
    println!("{:<20}{:>8}", "component", "params");
    for (name, count) in &r.counts {
        println!("{name:<20}{count:>8}");
    }
    println!("{:<20}{:>8}", "num_params()", r.num_params);
    println!("input [2, 128] -> logits {:?}", r.logits_dims);
    println!("logits[0, 0, :4]   = {:.4?}", r.first);
    println!("logits[1, 127, :4] = {:.4?}", r.last);
    println!(
        "cross-entropy      = {:.4}  (ln 62 = {:.4})",
        r.loss,
        62f32.ln()
    );
    println!(
        "change the last token: positions 0 to 126 move by {:e}, position 127 by {:.4}",
        r.causal.0, r.causal.1
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let r = build();
        // From python/main.py: the same weights, copied into nn.TransformerEncoder.
        let first = [-0.196566, 0.714586, 0.171288, -0.283295];
        let last = [-0.200070, 1.116862, 0.406603, -0.720139];
        assert_eq!(r.num_params, 116224);
        for i in 0..4 {
            assert!((r.first[i] - first[i]).abs() < 1e-4, "first {i}");
            assert!((r.last[i] - last[i]).abs() < 1e-4, "last {i}");
        }
        assert!((r.loss - 4.362418).abs() < 1e-4);
        assert!((r.causal.1 - 0.804828).abs() < 1e-4);
        assert_eq!(r.causal.0, 0.0);
    }
}
