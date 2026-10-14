use burn::tensor::activation::{gelu, log_softmax, softmax};
use burn::tensor::{Device, Int, Tensor, TensorData};

// A tiny GPT from raw tensors: no burn::nn, no Module, just Tensor<2> parameters
// in a Vec and the operations of Chapters 3, 5 and 10.
const V: usize = 62; // vocabulary: the distinct bytes of shakespeare.txt
const T: usize = 32; // context length (block size)
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
    fn below(&mut self, n: usize) -> usize {
        // two draws make 30 bits, enough for positions in a 200 KB text
        (((self.next() << 15) | self.next()) as usize) % n
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
    let qkv = x.matmul(w_qkv.clone()) + b_qkv.clone(); // [n*t, 3D]
    // [n*t, D] -> [n, t, H, dk] -> [n, H, t, dk]
    let heads = |m: Tensor<2>| m.reshape([n, t, H, dk]).swap_dims(1, 2);
    let q = heads(qkv.clone().slice([0..n * t, 0..D]));
    let k = heads(qkv.clone().slice([0..n * t, D..2 * D]));
    let v = heads(qkv.slice([0..n * t, 2 * D..3 * D]));
    let mask = Tensor::<2>::full([t, t], -1.0e9, &q.device())
        .triu(1)
        .reshape([1, 1, t, t]);
    let scores = q
        .matmul(k.swap_dims(2, 3))
        .mul_scalar(1.0 / (dk as f64).sqrt())
        + mask;
    let ctx = softmax(scores, 3).matmul(v); // [n, H, t, dk]
    let ctx = ctx.swap_dims(1, 2).reshape([n * t, D]); // glue the heads
    ctx.matmul(w_o.clone()) + b_o.clone()
}

// One pre-norm block (Chapter 10), with a GELU MLP.
fn block(x: Tensor<2>, p: &[Tensor<2>], n: usize, t: usize) -> Tensor<2> {
    let [ln1_g, ln1_b, .., ln2_g, ln2_b, w_fc, b_fc, w_pr, b_pr] = p else {
        unreachable!()
    };
    let x = x.clone() + attention(layer_norm(x, ln1_g, ln1_b), &p[2..6], n, t);
    let u = layer_norm(x.clone(), ln2_g, ln2_b);
    let h = gelu(u.matmul(w_fc.clone()) + b_fc.clone());
    x + h.matmul(w_pr.clone()) + b_pr.clone()
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
    x.matmul(p[0].clone().transpose()) // weight tying: the head is wte^T
}

// Mean next-token cross-entropy (Chapter 6) over all n*t positions.
fn cross_entropy(logits: Tensor<2>, targets: &[usize]) -> Tensor<1> {
    let dev = logits.device();
    let tg: Vec<i32> = targets.iter().map(|&i| i as i32).collect();
    let shape = [targets.len(), 1];
    let idx = Tensor::<2, Int>::from_data(TensorData::new(tg, shape), &dev);
    log_softmax(logits, 1).gather(1, idx).mean().neg()
}

struct Run {
    n_params: usize,
    logits_dims: [usize; 2],
    first_logits: Vec<f32>,
    loss: f32,
    past_change: f32,
    last_change: f32,
}

fn build() -> Run {
    let dev = Device::flex();
    let vocab = vocab();
    assert_eq!(vocab.len(), V);
    let data = encode(TEXT, &vocab);
    let p = init_params(42, &dev);
    let n_params: usize = p.iter().map(|t| t.dims()[0] * t.dims()[1]).sum();

    // Two windows of T + 1 tokens: inputs are the first T, targets the last T.
    let n = 2;
    let mut rng = Lcg(7);
    let starts: Vec<usize> = (0..n).map(|_| rng.below(data.len() - T - 1)).collect();
    let x: Vec<usize> = starts
        .iter()
        .flat_map(|&s| data[s..s + T].to_vec())
        .collect();
    let y: Vec<usize> = starts
        .iter()
        .flat_map(|&s| data[s + 1..s + T + 1].to_vec())
        .collect();
    let text: String = x[..T].iter().map(|&i| vocab[i] as char).collect();
    println!("input 0  = {text:?}");

    let logits = forward(&p, &x, n, T);
    let logits_dims = logits.dims();
    let loss: f32 = cross_entropy(logits.clone(), &y).into_scalar();
    let all: Vec<f32> = logits.into_data().try_to_vec().unwrap();

    // Causality: change the LAST input token of sequence 0. Only the logits at
    // that last position may move; positions 0..T-2 cannot see it.
    let mut x2 = x.clone();
    x2[T - 1] = (x2[T - 1] + 1) % V;
    let all2: Vec<f32> = forward(&p, &x2, n, T).into_data().try_to_vec().unwrap();
    let diff = |rows: std::ops::Range<usize>| {
        rows.flat_map(|r| (0..V).map(move |c| r * V + c))
            .map(|i| (all[i] - all2[i]).abs())
            .fold(0.0f32, f32::max)
    };
    Run {
        n_params,
        logits_dims,
        first_logits: all[..5].to_vec(),
        loss,
        past_change: diff(0..T - 1),
        last_change: diff(T - 1..T),
    }
}

fn main() {
    let run = build();
    println!(
        "parameters = {} in {} tensors",
        run.n_params,
        layout().len()
    );
    println!(
        "logits     = {:?}  (n*t rows, one per position)",
        run.logits_dims
    );
    println!("logits[0][..5] = {:.6?}", run.first_logits);
    println!(
        "loss       = {:.6}  (ln V = {:.6})",
        run.loss,
        (V as f32).ln()
    );
    println!(
        "change token 31 -> max|dlogits| at positions 0..30 = {:.1e}",
        run.past_change
    );
    println!(
        "                   max|dlogits| at position 31     = {:.4}",
        run.last_change
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let run = build();
        assert_eq!(run.n_params, 106112);
        assert_eq!(run.logits_dims, [64, 62]);
        // From python/main.py, the same weights and the same two windows.
        let torch_logits = [0.30312228, 0.39798650, -0.07456493, 0.32076013, 0.16374277];
        let torch_loss = 4.32309294;
        for (a, b) in run.first_logits.iter().zip(torch_logits) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        assert!((run.loss - torch_loss).abs() < 1e-5, "{}", run.loss);
        assert_eq!(run.past_change, 0.0);
        assert!(run.last_change > 0.01);
    }
}
