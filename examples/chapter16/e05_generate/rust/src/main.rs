use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderAutoregressiveCache, TransformerEncoderConfig,
    TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::{Device, Int, Tensor, TensorData};
use std::path::Path;
use std::time::Instant;

// Chapter 16 -- The generating program: config.json, vocab.txt and model.bpk in,
// Shakespeare out. It knows nothing about training.

// --- the model: Example 16.2's GPT ---
#[derive(Config, Debug)]
struct GptConfig {
    vocab_size: usize,
    #[config(default = 128)]
    block_size: usize,
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
                .with_norm_first(true)
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
    tok: Embedding,
    pos: Embedding,
    blocks: TransformerEncoder,
    ln_f: LayerNorm,
    head: Linear,
}

impl Gpt {
    fn embed(&self, tokens: Tensor<2, Int>) -> (TransformerEncoderInput, usize) {
        let [n, t] = tokens.dims();
        let device = tokens.device();
        let positions = Tensor::<1, Int>::arange(0..t as i64, &device).reshape([1, t]);
        let x = self.tok.forward(tokens) + self.pos.forward(positions);
        let causal = generate_autoregressive_mask(n, t, &device);
        (TransformerEncoderInput::new(x).mask_attn(causal), t)
    }

    // The logits of the LAST position only: [N, T] tokens in, [N, V] out.
    fn next_logits(&self, tokens: Tensor<2, Int>) -> Tensor<2> {
        let (input, t) = self.embed(tokens);
        self.last(self.blocks.forward(input), t)
    }

    // The same, with Burn's encoder cache: each call still passes the whole
    // sequence, but every sub-layer computes only the newest position and
    // appends it to its cache.
    fn next_logits_cached(
        &self,
        tokens: Tensor<2, Int>,
        cache: &mut TransformerEncoderAutoregressiveCache,
    ) -> Tensor<2> {
        let (input, t) = self.embed(tokens);
        self.last(
            self.blocks.forward_autoregressive_inference(input, cache),
            t,
        )
    }

    fn last(&self, x: Tensor<3>, t: usize) -> Tensor<2> {
        let [n, _, d] = x.dims();
        let x = x.slice([0..n, t - 1..t, 0..d]).reshape([n, d]);
        self.head.forward(self.ln_f.forward(x))
    }
}

// Example 16.2's fixed LCG weights: the parity test samples from the untrained model.
struct Lcg(u32);
#[allow(dead_code)] // uniform() and the fillers are used by the test only
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / 32768.0 // uniform on [0, 1), exact in f32
    }
    fn uniform(&mut self, n: usize, bound: f64) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() as f64 / 32768.0 * 2.0 - 1.0) * bound) as f32)
            .collect()
    }
    fn embedding(&mut self, e: &mut Embedding, rows: usize, d: usize, device: &Device) {
        e.weight = param(self.uniform(rows * d, 0.1), [rows, d], device);
    }
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

#[allow(dead_code)]
fn param<const D: usize>(v: Vec<f32>, shape: [usize; D], device: &Device) -> Param<Tensor<D>> {
    Param::from_tensor(Tensor::from_data(TensorData::new(v, shape), device))
}

#[allow(dead_code)]
impl Gpt {
    fn with_fixed_weights(mut self, seed: u32, device: &Device) -> Self {
        let [v, d] = self.tok.weight.dims();
        let t = self.pos.weight.dims()[0];
        let mut f = Lcg(seed);
        f.embedding(&mut self.tok, v, d, device);
        f.embedding(&mut self.pos, t, d, device);
        for layer in self.blocks.layers.iter_mut() {
            f.norm(&mut layer.norm_2, d, device);
            f.linear(&mut layer.mha.query, d, d, device);
            f.linear(&mut layer.mha.key, d, d, device);
            f.linear(&mut layer.mha.value, d, d, device);
            f.linear(&mut layer.mha.output, d, d, device);
            f.norm(&mut layer.norm_1, d, device);
            f.linear(&mut layer.pwff.linear_inner, d, 4 * d, device);
            f.linear(&mut layer.pwff.linear_outer, 4 * d, d, device);
        }
        f.norm(&mut self.ln_f, d, device);
        f.linear(&mut self.head, d, v, device);
        self
    }
}

// --- sampling ---
// One draw from a row of logits: divide by the temperature, keep the top k,
// softmax, and invert the cumulative distribution at u in [0, 1).
// Temperature 0 means greedy.
fn sample(logits: Tensor<1>, temperature: f64, top_k: usize, u: f32) -> u32 {
    if temperature == 0.0 {
        return logits.argmax(0).into_scalar::<i64>() as u32;
    }
    // The k largest scaled logits, largest first, and their token ids.
    let (top, ids) = logits.div_scalar(temperature).topk_with_indices(top_k, 0);
    let cdf = softmax(top, 0).cumsum(0);
    // The number of F_j below u is the (zero-based) position of the draw.
    let below = cdf.lower_scalar(u).int().sum().into_scalar::<i64>() as usize;
    let pick = below.min(top_k - 1); // guards against F_k = 0.99999994 < u
    ids.slice(pick..pick + 1).into_scalar::<i64>() as u32
}

// Generation: feed the last (at most) T tokens, draw one, append, repeat.
fn generate(
    model: &Gpt,
    prompt: &[u32],
    new_tokens: usize,
    temperature: f64,
    top_k: usize,
    seed: u32,
    device: &Device,
) -> Vec<u32> {
    let block = model.pos.weight.dims()[0];
    let mut rng = Lcg(seed);
    let mut ids = prompt.to_vec();
    let v = model.head.weight.dims()[1];
    for _ in 0..new_tokens {
        // Crop to the context length: the position table has only T rows.
        let context = &ids[ids.len().saturating_sub(block)..];
        let x = Tensor::<2, Int>::from_data(
            TensorData::new(
                context.iter().map(|&i| i as i64).collect(),
                [1, context.len()],
            ),
            device,
        );
        let logits = model.next_logits(x).reshape([v]);
        ids.push(sample(logits, temperature, top_k, rng.unit()));
    }
    ids
}

// The same loop through the cache. It only works while the sequence fits the context:
// a cropped window shifts every position, and the cache would be stale.
fn generate_cached(
    model: &Gpt,
    prompt: &[u32],
    new_tokens: usize,
    temperature: f64,
    top_k: usize,
    seed: u32,
    device: &Device,
) -> Vec<u32> {
    let block = model.pos.weight.dims()[0];
    assert!(prompt.len() + new_tokens <= block, "the cache cannot slide");
    let mut cache = model.blocks.new_autoregressive_cache();
    let mut rng = Lcg(seed);
    let mut ids = prompt.to_vec();
    let v = model.head.weight.dims()[1];
    for _ in 0..new_tokens {
        let x = Tensor::<2, Int>::from_data(
            TensorData::new(ids.iter().map(|&i| i as i64).collect(), [1, ids.len()]),
            device,
        );
        let logits = model.next_logits_cached(x, &mut cache).reshape([v]);
        ids.push(sample(logits, temperature, top_k, rng.unit()));
    }
    ids
}

// --- the generating program ---
#[derive(Config, Debug)]
struct SavedConfig {
    model: GptConfig, // the one part of Example 16.4's config.json a generator needs
}

struct Vocab(Vec<char>);
impl Vocab {
    fn encode(&self, s: &str) -> Vec<u32> {
        s.chars()
            .map(|c| self.0.iter().position(|&v| v == c).unwrap() as u32)
            .collect()
    }
    fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.0[i as usize]).collect()
    }
}

fn load(dir: &Path, device: &Device) -> (Gpt, Vocab) {
    let config = SavedConfig::load(dir.join("config.json"))
        .expect("config.json not found: run Example 16.4 first");
    let vocab = Vocab(
        std::fs::read_to_string(dir.join("vocab.txt"))
            .unwrap()
            .chars()
            .collect(),
    );
    // load_file adds the .bpk extension, and panics if the file does not fit the model.
    let model = config.model.init(device).load_file(dir.join("model"));
    (model, vocab)
}

fn main() {
    let device = Device::flex();
    device.seed(16); // nothing below draws from it; the LCG supplies every random number
    let dir = std::env::temp_dir().join("learning-burn-c16");
    let (model, vocab) = load(&dir, &device);
    println!(
        "loaded {} parameters, a vocabulary of {} symbols",
        model.num_params(),
        vocab.0.len()
    );

    let prompt = vocab.encode("MENENIUS:\n");
    let x = Tensor::<2, Int>::from_data(
        TensorData::new(
            prompt.iter().map(|&i| i as i64).collect(),
            [1, prompt.len()],
        ),
        &device,
    );
    let logits = model.next_logits(x).reshape([vocab.0.len()]);
    println!("next character after \"MENENIUS:\\n\", top 5:");
    for tau in [0.5, 1.0, 1.5] {
        let p = softmax(logits.clone().div_scalar(tau), 0);
        let (top, ids) = p.topk_with_indices(5, 0);
        let top: Vec<f32> = top.into_data().try_to_vec().unwrap();
        let ids: Vec<i64> = ids.into_data().iter::<i64>().collect(); // i32 on flex
        let row: Vec<String> = ids
            .iter()
            .zip(&top)
            .map(|(&i, p)| format!("{:?} {p:.3}", vocab.0[i as usize]))
            .collect();
        println!("  tau {tau:.1}: {}", row.join(", "));
    }

    for (tau, k) in [
        (0.0, 1),
        (0.8, 10),
        (1.0, vocab.0.len()),
        (1.5, vocab.0.len()),
    ] {
        let ids = generate(&model, &prompt, 160, tau, k, 2026, &device);
        println!("--- temperature {tau:.1}, top-k {k} ---");
        println!("{}", vocab.decode(&ids));
    }

    let n = model.pos.weight.dims()[0] - prompt.len(); // fill the context exactly
    let start = Instant::now();
    let plain = generate(&model, &prompt, n, 0.8, 10, 7, &device);
    let t_plain = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let cached = generate_cached(&model, &prompt, n, 0.8, 10, 7, &device);
    let t_cached = start.elapsed().as_secs_f64();
    println!(
        "{n} tokens with the encoder cache: identical = {}, {:.1}x faster",
        plain == cached,
        t_plain / t_cached
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // The untrained LCG model of Example 16.2, sampled with the LCG's numbers.
        let device = Device::flex();
        let model = GptConfig::new(62)
            .init(&device)
            .with_fixed_weights(42, &device);
        let prompt = [16, 44, 53, 54, 55]; // "First"
        let got = generate(&model, &prompt, 24, 0.8, 10, 7, &device);
        // From python/main.py: the same weights, logits and uniform numbers.
        let expected: [u32; 29] = [
            16, 44, 53, 54, 55, 18, 34, 1, 6, 1, 28, 60, 33, 21, 6, 6, 18, 17, 34, 28, 28, 61, 23,
            23, 28, 28, 34, 28, 15,
        ];
        assert_eq!(got, expected);
        assert_eq!(
            generate_cached(&model, &prompt, 24, 0.8, 10, 7, &device),
            got
        );
    }
}
