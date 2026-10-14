use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, DatasetError};
use burn::tensor::{Device, Int, Tensor, TensorData};
use std::sync::Arc;

// Chapter 16 -- A text dataset: characters in, fixed-length windows out.
//
// The corpus is a 200 KB slice of Tiny Shakespeare (public domain, via Karpathy's
// char-rnn), shipped in examples/chapter16/data/.

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/shakespeare.txt");
const BLOCK: usize = 128; // context length T
const BATCH: usize = 16; // batch size N

// --- the tokenizer: one token per character, ids in sorted order ---
#[derive(Clone, Debug)]
struct CharTokenizer {
    chars: Vec<char>,        // id -> character
    ids: [Option<u32>; 128], // ASCII character -> id
}

impl CharTokenizer {
    fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup(); // sorted, so ids never depend on the order of the text
        let mut ids = [None; 128];
        for (i, &c) in chars.iter().enumerate() {
            assert!(c.is_ascii(), "this tokenizer only handles ASCII");
            ids[c as usize] = Some(i as u32);
        }
        Self { chars, ids }
    }

    fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| self.ids[c as usize].expect("character not in the vocabulary"))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.chars[i as usize]).collect()
    }
}

// --- the dataset: window i is tokens[i*stride .. i*stride + T + 1] ---
#[derive(Clone, Debug)]
struct Window {
    start: usize,     // where the window begins in the token stream
    tokens: Vec<u32>, // T + 1 tokens: input = the first T, target = the last T
}

struct TextWindows {
    tokens: Arc<Vec<u32>>,
    block: usize,
    stride: usize,
}

impl TextWindows {
    fn new(tokens: Vec<u32>, block: usize, stride: usize) -> Self {
        Self {
            tokens: Arc::new(tokens),
            block,
            stride,
        }
    }
}

impl Dataset<Window> for TextWindows {
    fn get(&self, index: usize) -> Result<Window, DatasetError> {
        // An index past the end is a bug, not bad data: panic, as the trait asks.
        assert!(index < self.len(), "window {index} out of range");
        let start = index * self.stride;
        Ok(Window {
            start,
            tokens: self.tokens[start..start + self.block + 1].to_vec(),
        })
    }

    fn len(&self) -> usize {
        // every window needs T + 1 tokens: the last input token has a target too
        (self.tokens.len() - self.block - 1) / self.stride + 1
    }
}

// --- the batcher: inputs and targets are the same window, one token apart ---
#[derive(Clone, Debug)]
struct TextBatch {
    inputs: Tensor<2, Int>,  // [N, T]
    targets: Tensor<2, Int>, // [N, T]: targets[:, t] = inputs[:, t + 1]
}

#[derive(Clone, Default)]
struct TextBatcher;

impl Batcher<Window, TextBatch> for TextBatcher {
    fn batch(&self, items: Vec<Window>, device: &Device) -> TextBatch {
        let n = items.len();
        let t = items[0].tokens.len() - 1;
        let mut inputs = Vec::with_capacity(n * t);
        let mut targets = Vec::with_capacity(n * t);
        for w in &items {
            inputs.extend(w.tokens[..t].iter().map(|&id| id as i64));
            targets.extend(w.tokens[1..].iter().map(|&id| id as i64));
        }
        TextBatch {
            inputs: Tensor::from_data(TensorData::new(inputs, [n, t]), device),
            targets: Tensor::from_data(TensorData::new(targets, [n, t]), device),
        }
    }
}

// --- the split: the first 90% of the text trains, the last 10% validates ---
fn split(tokens: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let cut = tokens.len() * 9 / 10;
    (tokens[..cut].to_vec(), tokens[cut..].to_vec())
}

// Two baselines from counting: the entropy of the next character on its own, and given
// the character before it (unigram and bigram), in nats, over the training text.
fn entropies(tokens: &[u32], vocab: usize) -> (f64, f64) {
    let mut one = vec![0.0f64; vocab];
    let mut two = vec![0.0f64; vocab * vocab];
    for w in tokens.windows(2) {
        two[w[0] as usize * vocab + w[1] as usize] += 1.0;
    }
    for &t in tokens {
        one[t as usize] += 1.0;
    }
    let n = tokens.len() as f64;
    let h1 = -one
        .iter()
        .filter(|&&c| c > 0.0)
        .map(|c| c / n * (c / n).ln())
        .sum::<f64>();
    let pairs = n - 1.0;
    let h2 = -two
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0.0)
        .map(|(k, c)| c / pairs * (c / one[k / vocab]).ln())
        .sum::<f64>();
    (h1, h2)
}

// Only the window starts, to check what a shuffled epoch contains.
#[derive(Clone, Default)]
struct Starts;

impl Batcher<Window, Vec<usize>> for Starts {
    fn batch(&self, items: Vec<Window>, _device: &Device) -> Vec<usize> {
        items.iter().map(|w| w.start).collect()
    }
}

struct Summary {
    chars: usize,
    vocab_size: usize,
    vocab: String,
    encoded: Vec<u32>,
    round_trip: bool,
    split: (usize, usize),
    entropy: (f64, f64),
    windows: (usize, usize),
    window0: (String, String),
    batch_dims: ([usize; 2], [usize; 2]),
    shifted: bool,
    batch_sum: i64,
    epochs: (Vec<usize>, Vec<usize>, bool),
}

fn build() -> Summary {
    let device = Device::flex();
    let text = std::fs::read_to_string(DATA).expect("read shakespeare.txt");
    let tok = CharTokenizer::new(&text);
    let tokens = tok.encode(&text);
    let encoded = tok.encode("First Citizen:");
    let round_trip = tok.decode(&tokens) == text;

    let (train_ids, valid_ids) = split(&tokens);
    let split_sizes = (train_ids.len(), valid_ids.len());
    let entropy = entropies(&train_ids, tok.vocab_size());
    let train = TextWindows::new(train_ids, BLOCK, BLOCK);
    let valid = TextWindows::new(valid_ids, BLOCK, BLOCK);
    let windows = (train.len(), valid.len());

    let w0 = train.get(0).unwrap();
    let window0 = (tok.decode(&w0.tokens[..BLOCK]), tok.decode(&w0.tokens[1..]));

    // The first batch, in dataset order.
    let first = TextBatcher.batch(train.get_many((0..BATCH).collect()).unwrap(), &device);
    let batch_dims = (first.inputs.dims(), first.targets.dims());
    let shifted = first
        .inputs
        .clone()
        .slice([0..BATCH, 1..BLOCK])
        .equal(first.targets.clone().slice([0..BATCH, 0..BLOCK - 1]))
        .all()
        .into_scalar::<bool>();
    let batch_sum = first.inputs.sum().into_scalar::<i64>();

    // A loader that reshuffles the training windows every epoch, from one seed.
    let loader = DataLoaderBuilder::<Window, Vec<usize>>::new(Starts)
        .batch_size(BATCH)
        .shuffle(16)
        .set_device(device.clone())
        .build(train);
    let epoch1: Vec<usize> = loader.iter().flat_map(Result::unwrap).collect();
    let epoch2: Vec<usize> = loader.iter().flat_map(Result::unwrap).collect();
    let mut sorted = epoch1.clone();
    sorted.sort();
    let all_once = sorted == (0..windows.0).map(|i| i * BLOCK).collect::<Vec<_>>();

    Summary {
        chars: text.len(),
        vocab_size: tok.vocab_size(),
        vocab: tok.chars.iter().collect(),
        encoded,
        round_trip,
        split: split_sizes,
        entropy,
        windows,
        window0,
        batch_dims,
        shifted,
        batch_sum,
        epochs: (epoch1, epoch2, all_once),
    }
}

fn main() {
    let s = build();
    println!("text       = {} characters", s.chars);
    println!("vocabulary = {} symbols: {:?}", s.vocab_size, s.vocab);
    println!("encode(\"First Citizen:\") = {:?}", s.encoded);
    println!("decode(encode(text)) == text: {}", s.round_trip);
    println!(
        "split      = {} train / {} valid characters",
        s.split.0, s.split.1
    );
    println!(
        "entropy    = {:.4} nats per character alone, {:.4} given the one before",
        s.entropy.0, s.entropy.1
    );
    println!(
        "windows    = {} train / {} valid  (T = {BLOCK}, stride {BLOCK})",
        s.windows.0, s.windows.1
    );
    println!("window 0 input  = {:?}...", &s.window0.0[..32]);
    println!("window 0 target = {:?}...", &s.window0.1[..32]);
    println!(
        "first batch = inputs {:?}, targets {:?}, sum of input ids {}",
        s.batch_dims.0, s.batch_dims.1, s.batch_sum
    );
    println!("targets[:, t] == inputs[:, t + 1]: {}", s.shifted);
    println!("epoch 1 starts with windows at {:?}", &s.epochs.0[..4]);
    println!("epoch 2 starts with windows at {:?}", &s.epochs.1[..4]);
    println!(
        "each epoch sees every training window exactly once: {}",
        s.epochs.2
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let s = build();
        // From python/main.py: the same file, vocabulary, split and windows.
        assert_eq!(s.chars, 199994);
        assert_eq!(s.vocab_size, 62);
        assert_eq!(
            s.encoded,
            vec![16, 44, 53, 54, 55, 1, 13, 44, 55, 44, 61, 40, 49, 8]
        );
        assert!(s.round_trip);
        assert_eq!(s.split, (179994, 20000));
        assert!((s.entropy.0 - 3.297924).abs() < 1e-6);
        assert!((s.entropy.1 - 2.398677).abs() < 1e-6);
        assert_eq!(s.windows, (1406, 156));
        assert_eq!(s.batch_dims, ([16, 128], [16, 128]));
        assert!(s.shifted);
        assert_eq!(s.batch_sum, 73336);
        assert!(s.epochs.2);
        assert_ne!(s.epochs.0, s.epochs.1);
    }
}
