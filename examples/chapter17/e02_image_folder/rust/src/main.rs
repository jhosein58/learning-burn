use std::path::{Path, PathBuf};

use burn::backend::NdArray;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

type MyBackend = NdArray;

const ROOT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/shapes"
);

// A folder per class, an image per file:  shapes/1_vertical/03.png -> label 1.
// new() only LISTS the files. Nothing is decoded until get() asks for an item,
// so a folder of a million images costs a million paths in memory, not a million images.
struct ImageFolder {
    classes: Vec<String>,
    items: Vec<(PathBuf, usize)>,
}

impl ImageFolder {
    fn new(root: &Path) -> std::io::Result<Self> {
        let mut classes: Vec<String> = std::fs::read_dir(root)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        classes.sort(); // sorted, so label numbers never depend on the file system's order

        let mut items = Vec::new();
        for (label, class) in classes.iter().enumerate() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(root.join(class))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "png"))
                .collect();
            files.sort();
            items.extend(files.into_iter().map(|p| (p, label)));
        }
        Ok(Self { classes, items })
    }
}

#[derive(Clone, Debug)]
struct ImageItem {
    pixels: Vec<u8>, // 12 x 12 grayscale, row by row
    label: usize,
}

impl Dataset<ImageItem> for ImageFolder {
    fn get(&self, index: usize) -> Option<ImageItem> {
        let (path, label) = self.items.get(index)?;
        let img = image::open(path).ok()?.to_luma8(); // decode happens HERE
        Some(ImageItem {
            pixels: img.into_raw(),
            label: *label,
        })
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone, Default)]
struct ImageBatcher;

#[derive(Clone, Debug)]
struct ImageBatch<B: Backend> {
    images: Tensor<B, 4>,       // [batch, 1, 12, 12], values in [0, 1]
    targets: Tensor<B, 1, Int>, // [batch]
}

impl<B: Backend> Batcher<B, ImageItem, ImageBatch<B>> for ImageBatcher {
    fn batch(&self, items: Vec<ImageItem>, device: &B::Device) -> ImageBatch<B> {
        let n = items.len();
        let x: Vec<f32> = items
            .iter()
            .flat_map(|it| it.pixels.iter().map(|&p| p as f32 / 255.0))
            .collect();
        let y: Vec<i64> = items.iter().map(|it| it.label as i64).collect();
        ImageBatch {
            images: Tensor::from_data(TensorData::new(x, [n, 1, 12, 12]), device),
            targets: Tensor::from_data(TensorData::new(y, [n]), device),
        }
    }
}

/// Returns (classes, len, label of item 7, pixel sum of item 7, batch dims, batch labels, mean pixel).
fn build() -> (Vec<String>, usize, usize, u32, [usize; 4], Vec<i64>, f32) {
    let data = ImageFolder::new(Path::new(ROOT)).expect("list shapes/");
    let item = data.get(7).unwrap();
    let sum: u32 = item.pixels.iter().map(|&p| p as u32).sum();

    let device = Default::default();
    let first8: Vec<ImageItem> = (0..8).filter_map(|i| data.get(i)).collect();
    let batch: ImageBatch<MyBackend> = ImageBatcher.batch(first8, &device);
    let mean = batch.images.clone().mean().into_scalar();
    let labels: Vec<i64> = batch.targets.into_data().to_vec().unwrap();
    (
        data.classes.clone(),
        data.len(),
        item.label,
        sum,
        batch.images.dims(),
        labels,
        mean,
    )
}

fn main() {
    let (classes, len, label, sum, dims, labels, mean) = build();
    println!("classes     = {classes:?}");
    println!("images      = {len}");
    println!("item 7      = label {label}, pixel sum {sum}");
    println!("first batch = {dims:?}, labels {labels:?}, mean pixel {mean:.4}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (classes, len, label, sum, dims, labels, mean) = build();
        assert_eq!(
            classes,
            ["0_horizontal", "1_vertical", "2_diagonal", "3_plus"]
        );
        assert_eq!((len, label), (24, 1));
        assert_eq!(sum, 4017); // PIL reads the very same bytes
        assert_eq!(dims, [8, 1, 12, 12]);
        assert_eq!(labels, vec![0, 0, 0, 0, 0, 0, 1, 1]);
        assert!((mean - 0.110069).abs() < 1e-5);
    }
}
