use burn::backend::NdArray;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde::Deserialize;

type MyBackend = NdArray;

const CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/machines.csv"
);

// One row of the CSV. The field names match the header line; serde does the parsing
// and the type checking. A row with "hot" in the rpm column is an error, not a NaN.
#[derive(Deserialize, Debug, Clone)]
struct Reading {
    temperature: f32,
    vibration: f32,
    rpm: f32,
    failure: u8,
}

// Per-feature mean and standard deviation, computed once from the dataset.
#[derive(Clone, Debug)]
struct Standardize {
    mean: [f32; 3],
    std: [f32; 3],
}

impl Standardize {
    fn fit(data: &InMemDataset<Reading>) -> Self {
        let rows: Vec<[f32; 3]> = data
            .iter()
            .map(|r| [r.temperature, r.vibration, r.rpm])
            .collect();
        let n = rows.len() as f32;
        let mut mean = [0.0f32; 3];
        let mut std = [0.0f32; 3];
        for j in 0..3 {
            mean[j] = rows.iter().map(|r| r[j]).sum::<f32>() / n;
            let var = rows.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f32>() / n;
            std[j] = var.sqrt();
        }
        Self { mean, std }
    }
}

#[derive(Clone, Debug)]
struct MachineBatch<B: Backend> {
    features: Tensor<B, 2>,     // [batch, 3], standardised
    targets: Tensor<B, 1, Int>, // [batch]
}

// The Batcher is where rows become tensors, and where the scaling is applied.
impl<B: Backend> Batcher<B, Reading, MachineBatch<B>> for Standardize {
    fn batch(&self, items: Vec<Reading>, device: &B::Device) -> MachineBatch<B> {
        let n = items.len();
        let mut x = Vec::with_capacity(n * 3);
        for r in &items {
            for (j, v) in [r.temperature, r.vibration, r.rpm].into_iter().enumerate() {
                x.push((v - self.mean[j]) / self.std[j]);
            }
        }
        let y: Vec<i64> = items.iter().map(|r| r.failure as i64).collect();
        MachineBatch {
            features: Tensor::from_data(TensorData::new(x, [n, 3]), device),
            targets: Tensor::from_data(TensorData::new(y, [n]), device),
        }
    }
}

/// Returns (rows, failures, scaler, batch sizes, first standardised row).
fn build() -> (usize, usize, Standardize, Vec<usize>, Vec<f32>) {
    let data: InMemDataset<Reading> =
        InMemDataset::from_csv(CSV, &csv::ReaderBuilder::new()).expect("read machines.csv");
    let failures = data.iter().filter(|r| r.failure == 1).count();
    println!("first row  = {:?}", data.get(0).unwrap());

    let scaler = Standardize::fit(&data);
    let rows = data.len();
    let loader =
        DataLoaderBuilder::<MyBackend, Reading, MachineBatch<MyBackend>>::new(scaler.clone())
            .batch_size(32)
            .build(data);

    let mut sizes = Vec::new();
    let mut first = Vec::new();
    for batch in loader.iter() {
        let [n, _] = batch.features.dims();
        if sizes.is_empty() {
            println!(
                "batch      = features {:?}, targets {:?}",
                batch.features.dims(),
                batch.targets.dims()
            );
            first = batch
                .features
                .slice([0..1, 0..3])
                .into_data()
                .to_vec()
                .unwrap();
        }
        sizes.push(n);
    }
    (rows, failures, scaler, sizes, first)
}

fn main() {
    let (rows, failures, scaler, sizes, first) = build();
    println!("rows       = {rows} ({failures} failures)");
    println!("mean       = {:.3?}", scaler.mean);
    println!("std        = {:.3?}", scaler.std);
    println!("batches    = {sizes:?}");
    println!("row 0, standardised = {first:.4?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (rows, failures, scaler, sizes, first) = build();
        assert_eq!((rows, failures), (120, 42));
        assert_eq!(sizes, vec![32, 32, 32, 24]);
        // From python.py (numpy, float64, then compared at f32 precision).
        let mean = [66.823333, 5.334917, 2198.6083];
        let std = [14.729815, 2.487775, 793.63241];
        let row0 = [-1.488364, -0.625023, -1.367898];
        for j in 0..3 {
            assert!((scaler.mean[j] - mean[j]).abs() / mean[j].abs() < 1e-5);
            assert!((scaler.std[j] - std[j]).abs() / std[j].abs() < 1e-5);
            assert!((first[j] - row0[j]).abs() < 1e-4);
        }
    }
}
