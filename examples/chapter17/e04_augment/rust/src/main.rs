use std::sync::Arc;

use burn::data::dataset::transform::{ComposedDataset, Mapper, MapperDataset};
use burn::data::dataset::{Dataset, InMemDataset};

const ROOT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/shapes"
);
const SIZE: usize = 12;

#[derive(Clone, Debug)]
struct ImageItem {
    pixels: Vec<u8>, // 12 x 12, row by row
    label: usize,
}

// Example 17.2 does this lazily; 24 tiny images can simply be read up front.
fn load_all() -> InMemDataset<ImageItem> {
    let mut classes: Vec<_> = std::fs::read_dir(ROOT)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    classes.sort();
    let mut items = Vec::new();
    for (label, dir) in classes.iter().enumerate() {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        for f in files {
            let pixels = image::open(f).unwrap().to_luma8().into_raw();
            items.push(ImageItem { pixels, label });
        }
    }
    InMemDataset::new(items)
}

// A Mapper turns one item into another. MapperDataset applies it inside get(),
// so nothing is precomputed and nothing is stored twice.
#[derive(Clone, Copy)]
enum Augment {
    Identity,
    MirrorLeftRight,
}

impl Mapper<ImageItem, ImageItem> for Augment {
    fn map(&self, item: &ImageItem) -> ImageItem {
        match self {
            Augment::Identity => item.clone(),
            Augment::MirrorLeftRight => {
                let mut pixels = Vec::with_capacity(SIZE * SIZE);
                for row in item.pixels.chunks(SIZE) {
                    pixels.extend(row.iter().rev());
                }
                ImageItem {
                    pixels,
                    label: item.label,
                } // the label does not change
            }
        }
    }
}

fn row(item: &ImageItem, r: usize) -> Vec<u8> {
    item.pixels[r * SIZE..(r + 1) * SIZE].to_vec()
}

/// Returns (original len, augmented len, the bright row of image 0, the same row mirrored, round trip ok).
fn build() -> (usize, usize, Vec<u8>, Vec<u8>, bool) {
    let base = Arc::new(load_all());

    // Same Dataset TYPE for both halves (MapperDataset<_, Augment, _>), so they can be composed.
    let augmented = ComposedDataset::new(vec![
        MapperDataset::new(base.clone(), Augment::Identity),
        MapperDataset::new(base.clone(), Augment::MirrorLeftRight),
    ]);

    let original = augmented.get(0).unwrap(); // first half: untouched
    let mirrored = augmented.get(base.len()).unwrap(); // second half: the same image, mirrored

    // Image 0 is a horizontal bar: find its brightest row.
    let bright = (0..SIZE)
        .max_by_key(|&r| row(&original, r).iter().map(|&p| p as u32).sum::<u32>())
        .unwrap();
    let back = Augment::MirrorLeftRight.map(&mirrored);

    (
        base.len(),
        augmented.len(),
        row(&original, bright),
        row(&mirrored, bright),
        back.pixels == original.pixels && mirrored.label == original.label,
    )
}

fn main() {
    let (n, n_aug, before, after, round_trip) = build();
    println!("dataset: {n} images -> with mirrored copies: {n_aug}");
    println!("image 0, bright row          = {before:?}");
    println!("image {n}, same row, mirrored = {after:?}");
    println!("mirror twice == original, label kept: {round_trip}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pytorch() {
        let (n, n_aug, before, after, round_trip) = build();
        assert_eq!((n, n_aug), (24, 48));
        let expected: Vec<u8> = before.iter().rev().copied().collect(); // torch.flip(x, [-1])
        assert_eq!(after, expected);
        assert_eq!(
            before,
            vec![24, 32, 3, 18, 22, 231, 229, 239, 223, 251, 245, 17]
        );
        assert!(round_trip);
    }
}
