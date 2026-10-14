use burn::tensor::{Bool, Device, Tensor, TensorData};

// A bigram model is a table: row a holds the probabilities of the byte after a.
// "Training" is counting. Evaluating is a count-weighted sum of log-probabilities.
const TEXT: &[u8] = include_bytes!("../../../data/shakespeare.txt");

fn vocab() -> Vec<u8> {
    let mut seen = [false; 256];
    TEXT.iter().for_each(|&b| seen[b as usize] = true);
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}

// counts[a][b] = how often token b follows token a in `ids`.
fn count_pairs(ids: &[usize], v: usize, dev: &Device) -> Tensor<2> {
    let mut c = vec![0.0f32; v * v];
    ids.windows(2).for_each(|w| c[w[0] * v + w[1]] += 1.0);
    Tensor::from_data(TensorData::new(c, [v, v]), dev)
}

// Add-alpha smoothing: P[a][b] = (C[a][b] + alpha) / (sum_b C[a][b] + alpha V).
fn probabilities(counts: &Tensor<2>, alpha: f64) -> Tensor<2> {
    let v = counts.dims()[1] as f64;
    let rows = counts.clone().sum_dim(1); // [V, 1]
    counts.clone().add_scalar(alpha) / rows.add_scalar(alpha * v)
}

// Mean negative log-likelihood of the pairs counted in `eval`:
// NLL = -(1/M) sum_ab eval[a][b] log P[a][b],  M = number of pairs.
fn nll(eval: &Tensor<2>, probs: &Tensor<2>) -> f32 {
    let logp = probs.clone().log();
    let unused: Tensor<2, Bool> = eval.clone().equal_elem(0.0);
    // 0 * log 0 would be NaN; those terms are 0
    let logp = logp.mask_fill(unused, 0.0);
    let total: f32 = (eval.clone() * logp).sum().into_scalar();
    let m: f32 = eval.clone().sum().into_scalar();
    -total / m
}

fn show(b: u8) -> String {
    match b {
        b' ' => "_".to_string(),
        b'\n' => "\\n".to_string(),
        _ => (b as char).to_string(),
    }
}

struct Run {
    train_floor: f32,
    table: Vec<(f64, f32, f32)>, // (alpha, train NLL, val NLL)
    unseen: (usize, f32),
}

fn build(print: bool) -> Run {
    let dev = Device::flex();
    let vocab = vocab();
    let v = vocab.len();
    let ids: Vec<usize> = TEXT
        .iter()
        .map(|b| vocab.iter().position(|x| x == b).unwrap())
        .collect();
    let n_train = ids.len() * 9 / 10;
    let train = count_pairs(&ids[..n_train], v, &dev);
    let val = count_pairs(&ids[n_train..], v, &dev);

    // Pairs that occur in the validation text but never in training.
    let never: Tensor<2> = train.clone().equal_elem(0.0).float();
    let unseen_kinds: f32 = (never.clone() * val.clone().greater_elem(0.0).float())
        .sum()
        .into_scalar();
    let unseen_times: f32 = (never * val.clone()).sum().into_scalar();

    let mle = probabilities(&train, 0.0); // the counts themselves, normalised
    if print {
        let m: f32 = train.clone().sum().into_scalar();
        let mv: f32 = val.clone().sum().into_scalar();
        println!("vocab {v}, train pairs {m}, val pairs {mv}");
        for ctx in [b'q', b'T', b' '] {
            let a = vocab.iter().position(|&x| x == ctx).unwrap();
            let row = mle.clone().slice([a..a + 1, 0..v]);
            let (p, idx) = row.sort_descending_with_indices(1);
            let p: Vec<f32> = p.into_data().try_to_vec().unwrap();
            let idx: Vec<i32> = idx.into_data().try_to_vec().unwrap();
            let top: Vec<String> = (0..3)
                .filter(|&j| p[j] > 0.0)
                .map(|j| format!("{} {:.3}", show(vocab[idx[j] as usize]), p[j]))
                .collect();
            println!("after {:<2} -> {}", show(ctx), top.join(", "));
        }
    }
    let train_floor = nll(&train, &mle);
    let table = [1.0, 0.1, 0.01]
        .iter()
        .map(|&alpha| {
            let p = probabilities(&train, alpha);
            (alpha, nll(&train, &p), nll(&val, &p))
        })
        .collect();
    Run {
        train_floor,
        table,
        unseen: (unseen_kinds as usize, unseen_times),
    }
}

fn main() {
    let run = build(true);
    let (kinds, times) = run.unseen;
    println!("pairs seen in val but never in train: {kinds} kinds, {times} times");
    println!(
        "alpha 0 (pure counts): train NLL {:.4}, perplexity {:.2}; val NLL inf",
        run.train_floor,
        run.train_floor.exp()
    );
    println!("alpha  train NLL  val NLL  val perplexity");
    for (alpha, tr, va) in &run.table {
        println!("{alpha:<5}  {tr:.4}     {va:.4}   {:.2}", va.exp());
    }
    let v = 62f32;
    println!("uniform: NLL ln {v} = {:.4}, perplexity {v}", v.ln());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let run = build(false);
        // From python/main.py (torch), the same counts and the same formulas.
        assert_eq!(run.unseen, (17, 26.0));
        assert!((run.train_floor - 2.3986714).abs() < 1e-5);
        let torch = [
            (2.4122186, 2.4915168),
            (2.4001577, 2.4819686),
            (2.3988237, 2.4832907),
        ];
        for ((_, tr, va), (ttr, tva)) in run.table.iter().zip(torch) {
            assert!(
                (tr - ttr).abs() < 1e-5 && (va - tva).abs() < 1e-5,
                "{tr} {va}"
            );
        }
    }
}
