use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{Device, Int, Tensor, TensorData};

// The bigram model as a trainable table of logits W [V, V]: row a holds the
// scores of every byte that may follow a. Loss: softmax + cross-entropy.
// Gradient: (p - one_hot) on the rows that were used, derived by hand and
// checked against autodiff, then plain gradient descent on all the data.
const TEXT: &[u8] = include_bytes!("../../../data/shakespeare.txt");
const V: usize = 62;
const BATCH: usize = 8;
const LR: f64 = 100.0;
const STEPS: usize = 2000;

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) & 0x7FFF
    }
    fn uniform(&mut self) -> f32 {
        self.next() as f32 / 32768.0
    }
    fn below(&mut self, n: usize) -> usize {
        (((self.next() << 15) | self.next()) as usize) % n
    }
}

fn vocab() -> Vec<u8> {
    let mut seen = [false; 256];
    TEXT.iter().for_each(|&b| seen[b as usize] = true);
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}

fn count_pairs(ids: &[usize], dev: &Device) -> Tensor<2> {
    let mut c = vec![0.0f32; V * V];
    ids.windows(2).for_each(|w| c[w[0] * V + w[1]] += 1.0);
    Tensor::from_data(TensorData::new(c, [V, V]), dev)
}

fn ints<const R: usize>(v: &[usize], shape: [usize; R], dev: &Device) -> Tensor<R, Int> {
    let v: Vec<i32> = v.iter().map(|&i| i as i32).collect();
    Tensor::from_data(TensorData::new(v, shape), dev)
}

// Mean cross-entropy of a batch of pairs x_i -> y_i: row x_i, then -log p[y_i].
fn batch_loss(w: Tensor<2>, x: &[usize], y: &[usize]) -> Tensor<1> {
    let dev = w.device();
    let logits = w.select(0, ints(x, [x.len()], &dev)); // [N, V]
    log_softmax(logits, 1)
        .gather(1, ints(y, [y.len(), 1], &dev))
        .mean()
        .neg()
}

// The same loss over a whole text, through its pair counts C:
// L = -(1/M) sum_ab C[a][b] log softmax(W[a])[b].
fn count_loss(w: Tensor<2>, c: &Tensor<2>) -> Tensor<1> {
    let m: f32 = c.clone().sum().into_scalar();
    (c.clone() * log_softmax(w, 1)).sum().div_scalar(m).neg()
}

// By hand, batch form: dL/dW[x_i] += (softmax(W[x_i]) - one_hot(y_i)) / N.
fn batch_grad(w: &[f32], x: &[usize], y: &[usize]) -> Vec<f32> {
    let mut g = vec![0.0f32; V * V];
    for (&a, &b) in x.iter().zip(y) {
        let row = &w[a * V..(a + 1) * V];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|z| (z - m).exp()).collect();
        let s: f32 = e.iter().sum();
        for j in 0..V {
            let one_hot = if j == b { 1.0 } else { 0.0 };
            g[a * V + j] += (e[j] / s - one_hot) / x.len() as f32;
        }
    }
    g
}

// By hand, count form: dL/dW[a] = (n_a softmax(W[a]) - C[a]) / M,
// with n_a = sum_b C[a][b].
fn count_grad(w: Tensor<2>, c: &Tensor<2>) -> Tensor<2> {
    let m: f32 = c.clone().sum().into_scalar();
    let n_a = c.clone().sum_dim(1); // [V, 1]
    (softmax(w, 1) * n_a - c.clone()).div_scalar(m)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn vec(t: Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec().unwrap()
}

struct Run {
    pairs: String,
    batch_loss: f32,
    target_grad: f32,
    target_p: f32,
    batch_diff: f32,
    count_diff: f32,
    log: Vec<(usize, f32, f32)>,
    table: [(f32, f32); 2], // p(h | T) and p(u | q): trained, counted
}

fn build(steps: usize) -> Run {
    let dev = Device::flex().autodiff();
    let vocab = vocab();
    let ids: Vec<usize> = TEXT
        .iter()
        .map(|b| vocab.iter().position(|x| x == b).unwrap())
        .collect();
    let n_train = ids.len() * 9 / 10;
    let c_train = count_pairs(&ids[..n_train], &dev);
    let c_val = count_pairs(&ids[n_train..], &dev);

    let mut rng = Lcg(42);
    let w0: Vec<f32> = (0..V * V)
        .map(|_| (2.0 * rng.uniform() - 1.0) * 0.1)
        .collect();
    let w = Tensor::<2>::from_data(TensorData::new(w0.clone(), [V, V]), &dev).require_grad();

    // 1. A batch of 8 pairs: hand gradient against autodiff.
    let mut rng = Lcg(7);
    let starts: Vec<usize> = (0..BATCH).map(|_| rng.below(n_train - 1)).collect();
    let x: Vec<usize> = starts.iter().map(|&s| ids[s]).collect();
    let y: Vec<usize> = starts.iter().map(|&s| ids[s + 1]).collect();
    let show = |i: usize| match vocab[i] {
        b' ' => '_',
        b'\n' => '/',
        b => b as char,
    };
    let pairs = x
        .iter()
        .zip(&y)
        .map(|(&a, &b)| format!("{}{}", show(a), show(b)))
        .collect::<Vec<_>>()
        .join(" ");
    let loss = batch_loss(w.clone(), &x, &y);
    let auto = vec(w.grad(&loss.backward()).unwrap());
    let hand = batch_grad(&w0, &x, &y);
    let p0 = vec(softmax(w.clone().inner(), 1)); // the model's probabilities before training
    let batch_loss: f32 = loss.into_scalar();

    // 2. All 179,993 training pairs at once, through the counts.
    let loss = count_loss(w.clone(), &c_train);
    let auto_full = vec(w.grad(&loss.backward()).unwrap());
    let hand_full = vec(count_grad(w.clone().inner(), &c_train.clone().inner()));

    // 3. Plain gradient descent with the count-form gradient, no autodiff.
    let (c_train, c_val) = (c_train.inner(), c_val.inner());
    let mut w = w.inner();
    let mut log = Vec::new();
    for step in 0..=steps {
        if step % 500 == 0 {
            let tr: f32 = count_loss(w.clone(), &c_train).into_scalar();
            let va: f32 = count_loss(w.clone(), &c_val).into_scalar();
            log.push((step, tr, va));
        }
        if step < steps {
            w = w.clone() - count_grad(w, &c_train).mul_scalar(LR);
        }
    }
    // The trained table against the counts it should reproduce.
    let p = vec(softmax(w, 1));
    let counts = vec(c_train.clone() / c_train.sum_dim(1));
    let at = |a: u8, b: u8| {
        let pos = |c: u8| vocab.iter().position(|&x| x == c).unwrap();
        pos(a) * V + pos(b)
    };
    let table = [(b'T', b'h'), (b'q', b'u')].map(|(a, b)| (p[at(a, b)], counts[at(a, b)]));
    Run {
        pairs,
        batch_loss,
        target_grad: hand[x[0] * V + y[0]],
        target_p: p0[x[0] * V + y[0]],
        batch_diff: max_abs(&hand, &auto),
        count_diff: max_abs(&hand_full, &auto_full),
        log,
        table,
    }
}

fn main() {
    let run = build(STEPS);
    println!("batch of {BATCH} pairs: {}", run.pairs);
    println!("batch loss = {:.6}", run.batch_loss);
    println!(
        "hand dL/dW[x0][y0] = {:.6}  with p(y0 | x0) = {:.6}",
        run.target_grad, run.target_p
    );
    println!("batch form: max|hand - autodiff| = {:.1e}", run.batch_diff);
    println!("count form: max|hand - autodiff| = {:.1e}", run.count_diff);
    for (step, tr, va) in &run.log {
        println!(
            "step {step:>4}  train {tr:.4}  val {va:.4}  val perplexity {:.2}",
            va.exp()
        );
    }
    let [(th, th_c), (qu, qu_c)] = run.table;
    println!("p(h | T) = {th:.3} (counts {th_c:.3}),  p(u | q) = {qu:.3} (counts {qu_c:.3})");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let run = build(500);
        assert!(run.batch_diff < 1e-7 && run.count_diff < 1e-7);
        // From python/main.py: the same batch, the same weights, the same descent.
        assert!((run.batch_loss - 4.10793591).abs() < 1e-5);
        assert!((run.target_grad - -0.12059656).abs() < 1e-6);
        let torch = [(4.1233344, 4.1210017), (2.4120760, 2.5055213)];
        for ((_, tr, va), (ttr, tva)) in run.log.iter().zip(torch) {
            assert!(
                (tr - ttr).abs() < 1e-4 && (va - tva).abs() < 1e-4,
                "{tr} {va}"
            );
        }
    }
}
