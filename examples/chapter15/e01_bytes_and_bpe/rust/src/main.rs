use burn::tensor::{Device, Int, Tensor, TensorData};
use std::collections::HashMap;

// Tokenisers by hand: bytes as tokens, then byte-pair encoding (BPE).
const TEXT: &[u8] = include_bytes!("../../../data/shakespeare.txt");
const MERGES: usize = 48;
const SENTENCE: &str = "First Citizen:\nBefore we proceed any further, hear me speak.";

// The most frequent adjacent pair; ties go to the smallest pair, so that
// Rust and Python pick the same one.
fn most_frequent_pair(ids: &[u32]) -> ((u32, u32), usize) {
    let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
    for w in ids.windows(2) {
        *counts.entry((w[0], w[1])).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by(|(pa, ca), (pb, cb)| ca.cmp(cb).then(pb.cmp(pa)))
        .unwrap()
}

// Replace every occurrence of `pair`, left to right, by the new id.
fn merge(ids: &[u32], pair: (u32, u32), new: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
            out.push(new);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

struct Bpe {
    merges: Vec<((u32, u32), usize)>, // (pair, its count when it was merged)
    vocab: Vec<Vec<u8>>,              // id -> the bytes it stands for
    lengths: Vec<usize>,              // training-text length after 0, 1, ... merges
}

fn train(text: &[u8], n_merges: usize) -> Bpe {
    let mut ids: Vec<u32> = text.iter().map(|&b| b as u32).collect();
    let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let mut merges = Vec::new();
    let mut lengths = vec![ids.len()];
    for k in 0..n_merges {
        let (pair, count) = most_frequent_pair(&ids);
        let new = 256 + k as u32;
        ids = merge(&ids, pair, new);
        let bytes = [
            vocab[pair.0 as usize].clone(),
            vocab[pair.1 as usize].clone(),
        ]
        .concat();
        vocab.push(bytes);
        merges.push((pair, count));
        lengths.push(ids.len());
    }
    Bpe {
        merges,
        vocab,
        lengths,
    }
}

impl Bpe {
    // Encode with the first k merges, applied in the order they were learned.
    fn encode(&self, text: &str, k: usize) -> Vec<u32> {
        let mut ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        for (j, &(pair, _)) in self.merges[..k].iter().enumerate() {
            ids = merge(&ids, pair, 256 + j as u32);
        }
        ids
    }
    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.vocab[i as usize].clone())
            .collect();
        String::from_utf8(bytes).unwrap()
    }
    // Tokens joined by '|', with a space shown as '_' and a newline as \n.
    fn show(&self, ids: &[u32]) -> String {
        let piece = |i: &u32| self.decode(&[*i]).replace(' ', "_").replace('\n', "\\n");
        ids.iter().map(piece).collect::<Vec<_>>().join("|")
    }
}

fn main() {
    let used = {
        let mut seen = [false; 256];
        TEXT.iter().for_each(|&b| seen[b as usize] = true);
        seen.iter().filter(|&&s| s).count()
    };
    println!(
        "bytes: {} tokens, vocabulary 256 ({} used)",
        TEXT.len(),
        used
    );

    let bpe = train(TEXT, MERGES);
    for (k, &((a, b), count)) in bpe.merges[..12].iter().enumerate() {
        let new = 256 + k as u32;
        let (a, b, ab) = (bpe.show(&[a]), bpe.show(&[b]), bpe.show(&[new]));
        println!(
            "merge {:>2}: {a:>2} + {b:<2} -> {new} {ab:<3} ({count} times)",
            k + 1
        );
    }
    println!("merges  vocab  tokens  bytes/token");
    for k in [0, 12, 24, 48] {
        let n = bpe.lengths[k];
        println!(
            "{k:>6}  {:>5}  {n:>6}  {:.3}",
            256 + k,
            TEXT.len() as f64 / n as f64
        );
    }

    for k in [12, 48] {
        let ids = bpe.encode(SENTENCE, k);
        println!("{k:>2} merges, {} tokens: {}", ids.len(), bpe.show(&ids));
    }
    let ids = bpe.encode(SENTENCE, MERGES);
    println!("round trip: {}", bpe.decode(&ids) == SENTENCE);
    let unseen = bpe.encode("Rustacean", MERGES);
    println!(
        "unseen word: {} -> {:?}",
        bpe.show(&unseen),
        bpe.decode(&unseen)
    );

    // What the model receives: a Tensor of token ids.
    let device = Device::flex();
    let ints: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    let t = Tensor::<1, Int>::from_data(TensorData::new(ints, [ids.len()]), &device);
    println!("as a tensor: {:?} {}", t.dims(), t.slice([0..8]).to_data());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let bpe = train(TEXT, MERGES);
        // python/main.py learns the same merges in the same order.
        let first: Vec<(u32, u32)> = bpe.merges[..6].iter().map(|m| m.0).collect();
        assert_eq!(
            first,
            vec![
                (101, 32),
                (116, 104),
                (116, 32),
                (115, 32),
                (111, 117),
                (100, 32)
            ]
        );
        assert_eq!(bpe.lengths[MERGES], 138684);
        let ids = bpe.encode(SENTENCE, MERGES);
        assert_eq!(bpe.decode(&ids), SENTENCE);
        assert_eq!(bpe.decode(&bpe.encode("Rustacean", MERGES)), "Rustacean");
    }
}
