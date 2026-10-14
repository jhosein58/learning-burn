use burn::tensor::activation::{gelu, log_softmax, softmax};
use burn::tensor::{Device, Gradients, Int, Tensor, TensorData};

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

const N: usize = 16; // sequences per batch
const STEPS: usize = 2000;
const EVERY: usize = 400; // evaluate and print every EVERY steps
const LR: f64 = 3e-3;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8; // PyTorch's default, not Burn's 1e-5 (Chapter 7)

// N random windows of T + 1 tokens from `data`: inputs and next-token targets.
fn batch(data: &[usize], rng: &mut Lcg) -> (Vec<usize>, Vec<usize>) {
    let starts: Vec<usize> = (0..N).map(|_| rng.below(data.len() - T - 1)).collect();
    let x = starts
        .iter()
        .flat_map(|&s| data[s..s + T].to_vec())
        .collect();
    let y = starts
        .iter()
        .flat_map(|&s| data[s + 1..s + T + 1].to_vec())
        .collect();
    (x, y)
}

// Chapter 7's Adam, bias correction included, one tensor at a time.
// m and v live on the plain device: the optimizer state is never differentiated.
struct Adam {
    m: Vec<Tensor<2>>,
    v: Vec<Tensor<2>>,
    t: i32,
}

impl Adam {
    fn new(params: &[Tensor<2>]) -> Self {
        let zeros = || {
            params
                .iter()
                .map(|p| p.clone().inner().zeros_like())
                .collect()
        };
        Adam {
            m: zeros(),
            v: zeros(),
            t: 0,
        }
    }

    fn step(&mut self, p: Vec<Tensor<2>>, grads: &Gradients) -> Vec<Tensor<2>> {
        self.t += 1;
        let bc1 = 1.0 - BETA1.powi(self.t); // bias corrections
        let bc2 = 1.0 - BETA2.powi(self.t);
        let mut out = Vec::with_capacity(p.len());
        for (i, p) in p.into_iter().enumerate() {
            let g = p.grad(grads).unwrap();
            let (m, v) = (self.m[i].clone(), self.v[i].clone());
            let m = m.mul_scalar(BETA1) + g.clone().mul_scalar(1.0 - BETA1);
            let v = v.mul_scalar(BETA2) + g.powi_scalar(2).mul_scalar(1.0 - BETA2);
            let m_hat = m.clone().div_scalar(bc1);
            let v_hat = v.clone().div_scalar(bc2);
            let update = m_hat / v_hat.sqrt().add_scalar(EPS);
            (self.m[i], self.v[i]) = (m, v);
            // off autodiff for the arithmetic, back on as a fresh leaf
            let new = p.inner() - update.mul_scalar(LR);
            out.push(Tensor::from_inner(new).require_grad());
        }
        out
    }
}

// Mean loss over a fixed set of validation batches, without autodiff.
fn evaluate(p: &[Tensor<2>], val: &[(Vec<usize>, Vec<usize>)]) -> f32 {
    let plain: Vec<Tensor<2>> = p.iter().map(|t| t.clone().inner()).collect();
    let total: f32 = val
        .iter()
        .map(|(x, y)| cross_entropy(forward(&plain, x, N, T), y).into_scalar::<f32>())
        .sum();
    total / val.len() as f32
}

// Temperature sampling with the inverse CDF, one LCG uniform per token (Section 15.6).
fn generate(p: &[Tensor<2>], prompt: &[usize], new: usize, temp: f64, seed: u32) -> Vec<usize> {
    let plain: Vec<Tensor<2>> = p.iter().map(|t| t.clone().inner()).collect();
    let mut rng = Lcg(seed);
    let mut ids = prompt.to_vec();
    for _ in 0..new {
        let ctx = &ids[ids.len().saturating_sub(T)..]; // crop to the last T tokens
        let logits = forward(&plain, ctx, 1, ctx.len());
        let last = logits.slice([ctx.len() - 1..ctx.len(), 0..V]); // [1, V]
        let probs: Vec<f32> = softmax(last.div_scalar(temp), 1)
            .into_data()
            .try_to_vec()
            .unwrap();
        let u = rng.uniform();
        let mut cdf = 0.0;
        let next = probs
            .iter()
            .position(|&q| {
                cdf += q;
                u < cdf
            })
            .unwrap_or(V - 1);
        ids.push(next);
    }
    ids
}

struct Run {
    first_losses: Vec<f32>,
    log: Vec<(usize, f32, f32)>, // (step, mean train loss since the last entry, val loss)
    sample: String,
}

fn build(steps: usize) -> Run {
    let dev = Device::flex().autodiff();
    let vocab = vocab();
    let data = encode(TEXT, &vocab);
    let n_train = data.len() * 9 / 10;
    let (train, valid) = data.split_at(n_train);
    let mut vrng = Lcg(1);
    let val: Vec<_> = (0..4).map(|_| batch(valid, &mut vrng)).collect();

    let mut p: Vec<Tensor<2>> = init_params(42, &dev)
        .into_iter()
        .map(|t| t.require_grad())
        .collect();
    let mut adam = Adam::new(&p);
    let mut rng = Lcg(2024);
    let mut log = vec![(0, f32::NAN, evaluate(&p, &val))];
    let (mut first_losses, mut window) = (Vec::new(), 0.0f32);
    for step in 1..=steps {
        let (x, y) = batch(train, &mut rng);
        let loss = cross_entropy(forward(&p, &x, N, T), &y);
        let grads = loss.backward();
        let l: f32 = loss.into_scalar();
        p = adam.step(p, &grads);
        if step <= 3 {
            first_losses.push(l);
        }
        window += l;
        if step % EVERY == 0 {
            log.push((step, window / EVERY as f32, evaluate(&p, &val)));
            window = 0.0;
        }
    }
    let prompt = encode(b"CORIOLANUS:\n", &vocab);
    let ids = generate(&p, &prompt, 200, 0.8, 99);
    let sample = ids.iter().map(|&i| vocab[i] as char).collect();
    Run {
        first_losses,
        log,
        sample,
    }
}

fn main() {
    let start = std::time::Instant::now();
    let run = build(STEPS);
    println!("vocab {V}, context {T}, width {D}, {H} heads, {L} blocks, batch {N}x{T}");
    println!("steps 1-3 loss = {:.6?}", run.first_losses);
    for (step, train, val) in &run.log {
        if step == &0 {
            println!("step {step:>4}  val {val:.4}");
        } else {
            println!("step {step:>4}  train {train:.4}  val {val:.4}");
        }
    }
    println!("--- sample (temperature 0.8) ---\n{}", run.sample);
    eprintln!("time = {:.1} s", start.elapsed().as_secs_f32());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        // Same LCG weights, same batches, same hand-written Adam as python/main.py:
        // the first three losses must agree.
        let run = build(3);
        let torch = [4.28967810, 3.76154423, 3.68797398];
        for (a, b) in run.first_losses.iter().zip(torch) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
