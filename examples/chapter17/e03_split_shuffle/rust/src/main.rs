use std::sync::Arc;

use burn::backend::NdArray;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::transform::{PartialDataset, ShuffledDataset};
use burn::data::dataset::{Dataset, InMemDataset};
use burn::tensor::backend::Backend;

type MyBackend = NdArray;

// Items are just their own index, 0..20, so every order is easy to read.
#[derive(Clone, Default)]
struct Ids;

impl<B: Backend> Batcher<B, u32, Vec<u32>> for Ids {
    fn batch(&self, items: Vec<u32>, _device: &B::Device) -> Vec<u32> {
        items
    }
}

fn is_permutation(mut seen: Vec<u32>, of: &[u32]) -> bool {
    let mut want = of.to_vec();
    seen.sort();
    want.sort();
    seen == want
}

/// Returns (train ids, val ids, epoch-1 order, epoch-2 order, epoch-1 order of a rebuilt loader).
fn build() -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let all = InMemDataset::new((0..20u32).collect::<Vec<_>>());

    // 1. Shuffle ONCE with a fixed seed, then cut: 16 for training, 4 for validation.
    //    Shuffle before splitting, or the validation set is just "the last rows of the file".
    let shuffled = Arc::new(ShuffledDataset::new(all, 42));
    let train = PartialDataset::new(shuffled.clone(), 0, 16);
    let val = PartialDataset::new(shuffled, 16, 20);
    let train_ids: Vec<u32> = train.iter().collect();
    let val_ids: Vec<u32> = val.iter().collect();

    // 2. A loader that reshuffles the training set EVERY epoch, from one seed.
    let make_loader = || {
        DataLoaderBuilder::<MyBackend, u32, Vec<u32>>::new(Ids)
            .batch_size(4)
            .shuffle(7)
            .build(InMemDataset::new(train_ids.clone()))
    };
    let loader = make_loader();
    let epoch1: Vec<u32> = loader.iter().flatten().collect();
    let epoch2: Vec<u32> = loader.iter().flatten().collect();
    let again: Vec<u32> = make_loader().iter().flatten().collect(); // same seed, fresh loader

    (train_ids, val_ids, epoch1, epoch2, again)
}

fn main() {
    let (train, val, epoch1, epoch2, again) = build();
    println!("train ({}) = {train:?}", train.len());
    println!("val   ({})  = {val:?}", val.len());
    println!("epoch 1    = {epoch1:?}");
    println!("epoch 2    = {epoch2:?}");
    println!(
        "epoch 1 of a rebuilt loader, same seed = {}",
        if again == epoch1 {
            "identical"
        } else {
            "DIFFERENT"
        }
    );
    println!(
        "each epoch sees every training item exactly once: {}",
        is_permutation(epoch1.clone(), &train) && is_permutation(epoch2.clone(), &train)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_guarantees_as_pytorch() {
        // The RNGs differ, so the ORDERS differ from python.py. The guarantees must not.
        let (train, val, epoch1, epoch2, again) = build();
        assert_eq!((train.len(), val.len()), (16, 4));
        let mut all: Vec<u32> = train.iter().chain(&val).copied().collect();
        all.sort();
        assert_eq!(all, (0..20).collect::<Vec<u32>>()); // disjoint, and nothing lost
        assert!(is_permutation(epoch1.clone(), &train));
        assert!(is_permutation(epoch2.clone(), &train));
        assert_ne!(epoch1, epoch2); // a new order every epoch
        assert_eq!(epoch1, again); //  ...but reproducible from the seed
    }
}
