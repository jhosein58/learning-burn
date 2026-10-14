# Example 16.1 — A Text Dataset

A character tokeniser over the Shakespeare slice (62 symbols, ids in sorted order), a contiguous 90/10 train/validation split of the text, a `Dataset` of fixed-length windows (`T + 1 = 129` tokens each, stride 128), and a `Batcher` that turns windows into `[N, T]` `Int` inputs and targets shifted by one token. The training loader reshuffles the windows every epoch from one seed.

```rust
impl Dataset<Window> for TextWindows {
    fn get(&self, index: usize) -> Result<Window, DatasetError> { /* tokens[i*stride ..= i*stride + T] */ }
    fn len(&self) -> usize { (self.tokens.len() - self.block - 1) / self.stride + 1 }
}
```

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Burn version + `#[test]`. |
| `python/main.py` | PyTorch reference + `pytest` test. |
| `../data/shakespeare.txt` | The corpus (Shakespeare, public domain, via Karpathy's char-rnn). |

## Run & test

```bash
cargo run -- -c 16 -e 1 rust      # from the repo root
cargo run -- -c 16 -e 1 python
cargo test -p c16e1 --release     # the Rust parity test
pytest examples/chapter16/e01_text_dataset/python/main.py      # the Python parity test
```

## Parity

Both sides build the same vocabulary, encode `"First Citizen:"` to the same ids, cut 179,994 / 20,000 characters, make 1,406 / 156 windows and the same first batch (sum of ids 73,336), and count the same unigram and bigram entropies of the training text (3.2979 and 2.3987 nats) to 1e-6. The shuffled orders differ, because the two frameworks use different random generators; both tests check what a shuffle promises instead (every window exactly once per epoch, a new order each epoch).
