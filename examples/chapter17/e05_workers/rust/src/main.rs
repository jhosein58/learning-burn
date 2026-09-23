use std::time::{Duration, Instant};

use burn::backend::NdArray;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;

type MyBackend = NdArray;

// A dataset whose get() is slow on purpose: 4 ms per item stands in for
// opening a file and decoding a JPEG. Real datasets are slow in exactly this place.
struct SlowDataset {
    n: usize,
}

impl Dataset<usize> for SlowDataset {
    fn get(&self, index: usize) -> Option<usize> {
        if index >= self.n {
            return None;
        }
        std::thread::sleep(Duration::from_millis(4));
        Some(index)
    }

    fn len(&self) -> usize {
        self.n
    }
}

#[derive(Clone, Default)]
struct Ids;

impl<B: Backend> Batcher<B, usize, Vec<usize>> for Ids {
    fn batch(&self, items: Vec<usize>, _device: &B::Device) -> Vec<usize> {
        items
    }
}

/// One full pass over 64 items in batches of 8. Returns (items seen, sorted; elapsed).
fn one_epoch(workers: usize) -> (Vec<usize>, Duration) {
    let mut builder = DataLoaderBuilder::<MyBackend, usize, Vec<usize>>::new(Ids).batch_size(8);
    if workers > 0 {
        builder = builder.num_workers(workers); // each worker thread owns a slice of the dataset
    }
    let loader = builder.build(SlowDataset { n: 64 });

    let start = Instant::now();
    let mut seen: Vec<usize> = loader.iter().flatten().collect();
    let elapsed = start.elapsed();
    seen.sort();
    (seen, elapsed)
}

fn build() -> Vec<(usize, Vec<usize>, Duration)> {
    [0, 2, 4]
        .into_iter()
        .map(|w| {
            let (seen, t) = one_epoch(w);
            (w, seen, t)
        })
        .collect()
}

fn main() {
    let runs = build();
    let base = runs[0].2.as_secs_f64();
    for (workers, seen, t) in &runs {
        let label = if *workers == 0 {
            "no workers".to_string()
        } else {
            format!("{workers} workers")
        };
        println!(
            "{label:<10}: {} items in {:>4} ms  ({:>5.0} items/s, {:.1}x)",
            seen.len(),
            t.as_millis(),
            seen.len() as f64 / t.as_secs_f64(),
            base / t.as_secs_f64()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_guarantees_as_pytorch() {
        // Timings vary from machine to machine; what must not vary is that every
        // item arrives exactly once, whatever the number of workers.
        for (_, seen, _) in build() {
            assert_eq!(seen, (0..64).collect::<Vec<usize>>());
        }
    }
}
