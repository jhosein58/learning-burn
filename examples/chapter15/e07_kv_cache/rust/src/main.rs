use burn::tensor::activation::{gelu, softmax};
use burn::tensor::{Device, Int, Tensor, TensorData};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// Example 15.4's GPT with a longer context, generating greedily twice:
// once by re-running the whole prefix every step, once with a KV cache.
// Every matmul goes through mm(), which counts its multiply-adds.
const V: usize = 62; // vocabulary: the distinct bytes of shakespeare.txt
const T: usize = 128; // context length (block size)
const D: usize = 64; // model width
const H: usize = 4; // attention heads, each D / H = 16 wide
const L: usize = 2; // transformer blocks
const FF: usize = 4 * D; // MLP hidden width

const TEXT: &[u8] = include_bytes!("../../../data/shakespeare.txt");

// The LCG of Chapters 11-14, bit-identical to python/main.py.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF // 15 bits
    }
    fn uniform(&mut self) -> f32 {
        self.next() as f32 / 32768.0 // [0, 1), exact in f32
    }
}

// Token ids: each distinct byte of the text, in sorted order, gets an id.
fn vocab() -> Vec<u8> {
    let mut seen = [false; 256];
    TEXT.iter().for_each(|&b| seen[b as usize] = true);
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}

fn encode(text: &[u8], vocab: &[u8]) -> Vec<usize> {
    text.iter()
        .map(|b| vocab.iter().position(|v| v == b).unwrap())
        .collect()
}

enum Init {
    Uniform(f32), // (2u - 1) * scale
    Zeros,
    Ones,
}

// Every parameter, in a fixed order, with its shape and its initialisation.
// Biases and LayerNorm gains are [1, D] rows so that everything is a Tensor<2>.
fn layout() -> Vec<(String, [usize; 2], Init)> {
    let s = 1.0 / (D as f32).sqrt(); // 1/sqrt(fan_in) for D-wide inputs
    let sf = 1.0 / (FF as f32).sqrt(); // ... and for the MLP's second matrix
    let mut p = vec![
        ("wte".to_string(), [V, D], Init::Uniform(0.1)),
        ("wpe".to_string(), [T, D], Init::Uniform(0.1)),
    ];
    for l in 0..L {
        let blk = [
            ("ln1.g", [1, D], Init::Ones),
            ("ln1.b", [1, D], Init::Zeros),
            ("attn.w_qkv", [D, 3 * D], Init::Uniform(s)),
            ("attn.b_qkv", [1, 3 * D], Init::Zeros),
            ("attn.w_o", [D, D], Init::Uniform(s)),
            ("attn.b_o", [1, D], Init::Zeros),
            ("ln2.g", [1, D], Init::Ones),
            ("ln2.b", [1, D], Init::Zeros),
            ("mlp.w_fc", [D, FF], Init::Uniform(s)),
            ("mlp.b_fc", [1, FF], Init::Zeros),
            ("mlp.w_pr", [FF, D], Init::Uniform(sf)),
            ("mlp.b_pr", [1, D], Init::Zeros),
        ];
        p.extend(blk.map(|(n, sh, i)| (format!("h{l}.{n}"), sh, i)));
    }
    p.push(("lnf.g".to_string(), [1, D], Init::Ones));
    p.push(("lnf.b".to_string(), [1, D], Init::Zeros));
    p
}

fn init_params(seed: u32, dev: &Device) -> Vec<Tensor<2>> {
    let mut rng = Lcg(seed);
    layout()
        .into_iter()
        .map(|(_, [r, c], init)| {
            let vals: Vec<f32> = (0..r * c)
                .map(|_| match init {
                    Init::Uniform(s) => (2.0 * rng.uniform() - 1.0) * s,
                    Init::Zeros => 0.0,
                    Init::Ones => 1.0,
                })
                .collect();
            Tensor::from_data(TensorData::new(vals, [r, c]), dev)
        })
        .collect()
}

// Multiply-add counter: [.., m, k] x [.., k, n] costs (batch * m * k) * n.
static MACS: AtomicU64 = AtomicU64::new(0);
fn mm<const R: usize>(a: Tensor<R>, b: Tensor<R>) -> Tensor<R> {
    let n = b.dims()[R - 1];
    let lhs: usize = a.dims().iter().product();
    MACS.fetch_add((lhs * n) as u64, Ordering::Relaxed);
    a.matmul(b)
}

// LayerNorm (Chapter 10), with its learned gain and shift.
fn layer_norm(x: Tensor<2>, g: &Tensor<2>, b: &Tensor<2>) -> Tensor<2> {
    let mean = x.clone().mean_dim(1); // [rows, 1]
    let xc = x - mean;
    let var = xc.clone().powi_scalar(2).mean_dim(1); // biased variance
    xc / var.add_scalar(1e-5).sqrt() * g.clone() + b.clone()
}

// Causal multi-head self-attention, Chapter 10's Examples 10.2 and 10.3 batched.
// x is [n*t, D]; the result is [n*t, D].
fn attention(x: Tensor<2>, p: &[Tensor<2>], n: usize, t: usize) -> Tensor<2> {
    let [w_qkv, b_qkv, w_o, b_o] = p else {
        unreachable!()
    };
    let dk = D / H;
    let qkv = mm(x, w_qkv.clone()) + b_qkv.clone(); // [n*t, 3D]
    // [n*t, D] -> [n, t, H, dk] -> [n, H, t, dk]
    let heads = |m: Tensor<2>| m.reshape([n, t, H, dk]).swap_dims(1, 2);
    let q = heads(qkv.clone().slice([0..n * t, 0..D]));
    let k = heads(qkv.clone().slice([0..n * t, D..2 * D]));
    let v = heads(qkv.slice([0..n * t, 2 * D..3 * D]));
    let mask = Tensor::<2>::full([t, t], -1.0e9, &q.device())
        .triu(1)
        .reshape([1, 1, t, t]);
    let scores = mm(q, k.swap_dims(2, 3)).mul_scalar(1.0 / (dk as f64).sqrt()) + mask;
    let ctx = mm(softmax(scores, 3), v); // [n, H, t, dk]
    let ctx = ctx.swap_dims(1, 2).reshape([n * t, D]); // glue the heads
    mm(ctx, w_o.clone()) + b_o.clone()
}

// One pre-norm block (Chapter 10), with a GELU MLP.
fn block(x: Tensor<2>, p: &[Tensor<2>], n: usize, t: usize) -> Tensor<2> {
    let [ln1_g, ln1_b, .., ln2_g, ln2_b, w_fc, b_fc, w_pr, b_pr] = p else {
        unreachable!()
    };
    let x = x.clone() + attention(layer_norm(x, ln1_g, ln1_b), &p[2..6], n, t);
    let u = layer_norm(x.clone(), ln2_g, ln2_b);
    let h = gelu(mm(u, w_fc.clone()) + b_fc.clone());
    x + mm(h, w_pr.clone()) + b_pr.clone()
}

// ids: n sequences of t token ids, flattened. Returns logits [n*t, V].
fn forward(p: &[Tensor<2>], ids: &[usize], n: usize, t: usize) -> Tensor<2> {
    let dev = p[0].device();
    let ids: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    let idx = Tensor::<1, Int>::from_data(TensorData::new(ids, [n * t]), &dev);
    let tok = p[0].clone().select(0, idx); // embedding = row lookup, [n*t, D]
    let pos = p[1].clone().slice([0..t, 0..D]); // [t, D]
    let x = tok.reshape([n, t, D]) + pos.unsqueeze::<3>(); // [n, t, D]
    let mut x = x.reshape([n * t, D]);
    for l in 0..L {
        x = block(x, &p[2 + 12 * l..2 + 12 * (l + 1)], n, t);
    }
    let x = layer_norm(x, &p[2 + 12 * L], &p[3 + 12 * L]);
    mm(x, p[0].clone().transpose()) // weight tying: the head is wte^T
}

// The KV cache: for every block, the keys and values of all positions so far.
struct KvCache {
    k: Vec<Option<Tensor<2>>>, // per block, [t, D]
    v: Vec<Option<Tensor<2>>>,
}

// Append one row to a cached [t, D] tensor (or start it).
fn append(cache: &mut Option<Tensor<2>>, row: Tensor<2>) -> Tensor<2> {
    let all = match cache.take() {
        Some(old) => Tensor::cat(vec![old, row], 0),
        None => row,
    };
    *cache = Some(all.clone());
    all
}

// One new token at position `pos`: only its own query, key and value are
// computed; the keys and values of earlier positions come from the cache.
// No mask is needed: the cache only holds the past.
fn step(p: &[Tensor<2>], kv: &mut KvCache, id: usize, pos: usize) -> Tensor<2> {
    let dev = p[0].device();
    let dk = D / H;
    let scale = 1.0 / (dk as f64).sqrt();
    let idx = TensorData::new(vec![id as i32], [1]);
    let tok = p[0]
        .clone()
        .select(0, Tensor::<1, Int>::from_data(idx, &dev));
    let mut x = tok + p[1].clone().slice([pos..pos + 1, 0..D]); // [1, D]
    for l in 0..L {
        let b = &p[2 + 12 * l..2 + 12 * (l + 1)];
        let [ln1_g, ln1_b, w_qkv, b_qkv, w_o, b_o] = &b[..6] else {
            unreachable!()
        };
        let [ln2_g, ln2_b, w_fc, b_fc, w_pr, b_pr] = &b[6..] else {
            unreachable!()
        };
        let u = layer_norm(x.clone(), ln1_g, ln1_b);
        let qkv = mm(u, w_qkv.clone()) + b_qkv.clone(); // [1, 3D]
        let q = qkv.clone().slice([0..1, 0..D]);
        // append this position's key and value: [t, D] each
        let k = append(&mut kv.k[l], qkv.clone().slice([0..1, D..2 * D]));
        let v = append(&mut kv.v[l], qkv.slice([0..1, 2 * D..3 * D]));
        let t = k.dims()[0];
        let q = q.reshape([1, H, dk]).swap_dims(0, 1); // [H, 1, dk]
        let k = k.reshape([t, H, dk]).swap_dims(0, 1); // [H, t, dk]
        let v = v.reshape([t, H, dk]).swap_dims(0, 1);
        let scores = mm(q, k.swap_dims(1, 2)).mul_scalar(scale); // [H, 1, t]
        let ctx = mm(softmax(scores, 2), v).swap_dims(0, 1).reshape([1, D]);
        x = x + mm(ctx, w_o.clone()) + b_o.clone();
        let u = layer_norm(x.clone(), ln2_g, ln2_b);
        let h = gelu(mm(u, w_fc.clone()) + b_fc.clone());
        x = x + mm(h, w_pr.clone()) + b_pr.clone();
    }
    let x = layer_norm(x, &p[2 + 12 * L], &p[3 + 12 * L]);
    mm(x, p[0].clone().transpose()) // [1, V]
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

struct Gen {
    ids: Vec<usize>,
    logits: Vec<Vec<f32>>, // the logits that chose each new token
    macs: u64,
    ms: f64,
}

// Without a cache: every step runs the whole prefix and keeps the last row.
fn generate_full(p: &[Tensor<2>], prompt: &[usize], new: usize) -> Gen {
    MACS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let (mut ids, mut logits) = (prompt.to_vec(), Vec::new());
    for _ in 0..new {
        let t = ids.len();
        let all = forward(p, &ids, 1, t);
        let last: Vec<f32> = all
            .slice([t - 1..t, 0..V])
            .into_data()
            .try_to_vec()
            .unwrap();
        ids.push(argmax(&last));
        logits.push(last);
    }
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let macs = MACS.load(Ordering::Relaxed);
    Gen {
        ids,
        logits,
        macs,
        ms,
    }
}

// With a cache: feed each token once, prompt included.
fn generate_cached(p: &[Tensor<2>], prompt: &[usize], new: usize) -> Gen {
    MACS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let mut cache = KvCache {
        k: vec![None; L],
        v: vec![None; L],
    };
    let (mut ids, mut logits) = (prompt.to_vec(), Vec::new());
    let mut last = Vec::new();
    for (pos, &id) in prompt.iter().enumerate() {
        last = step(p, &mut cache, id, pos)
            .into_data()
            .try_to_vec()
            .unwrap();
    }
    for _ in 0..new {
        let next = argmax(&last);
        ids.push(next);
        logits.push(last);
        if ids.len() == prompt.len() + new {
            break; // the last token is chosen; it never needs to be fed
        }
        last = step(p, &mut cache, next, ids.len() - 1)
            .into_data()
            .try_to_vec()
            .unwrap();
    }
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let macs = MACS.load(Ordering::Relaxed);
    Gen {
        ids,
        logits,
        macs,
        ms,
    }
}

fn build() -> (Gen, Gen, Vec<u8>, usize) {
    let dev = Device::flex();
    let vocab = vocab();
    let p = init_params(42, &dev);
    let prompt = encode(b"CORIOLANUS:\n", &vocab);
    let new = T - prompt.len();
    let full = generate_full(&p, &prompt, new);
    let cached = generate_cached(&p, &prompt, new);
    (full, cached, vocab, prompt.len())
}

fn max_diff(a: &Gen, b: &Gen) -> f32 {
    a.logits
        .iter()
        .flatten()
        .zip(b.logits.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() {
    let (full, cached, vocab, n_prompt) = build();
    let new = full.ids.len() - n_prompt;
    println!("context {T}, prompt {n_prompt} tokens, {new} new tokens, greedy");
    println!("          multiply-adds      time");
    for (name, g) in [("no cache", &full), ("KV cache", &cached)] {
        println!("{name}  {:>13}  {:>6.1} ms", g.macs, g.ms);
    }
    let ratio = full.macs as f64 / cached.macs as f64;
    println!("ratio     {ratio:>12.1}x  {:>8.1}x", full.ms / cached.ms);
    println!(
        "max|logits(no cache) - logits(cache)| over {new} steps = {:.1e}",
        max_diff(&full, &cached)
    );
    println!("same tokens: {}", full.ids == cached.ids);
    let text: String = cached.ids[n_prompt..]
        .iter()
        .map(|&i| vocab[i] as char)
        .collect();
    println!("generated (untrained weights): {:?}", &text[..20]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (full, cached, _, _) = build();
        assert!(max_diff(&full, &cached) < 1e-4);
        assert_eq!(full.ids, cached.ids);
        // From python/main.py: the same counts, tokens and final logits.
        assert_eq!((full.macs, cached.macs), (1001252608, 15069312));
        let torch_ids: Vec<usize> = vec![28; 10];
        assert_eq!(&full.ids[12..22], &torch_ids[..]);
        let torch_last = [0.3557437, 0.4652831, -0.2058321, 0.1842696, 0.5877394];
        for (a, b) in full.logits.last().unwrap().iter().zip(torch_last) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
