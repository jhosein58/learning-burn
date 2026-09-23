"""Example 17.4 - Augmentation as a Lazy Mapping (PyTorch reference + parity test).

Mirror every image left-to-right and concatenate the mirrored copies with the
originals: 24 images become 48, and nothing is stored twice.

Run:   python main.py
Test:  pytest main.py
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "shapes")


def load_all():
    items = []
    for label, c in enumerate(sorted(os.listdir(ROOT))):
        for f in sorted(os.listdir(os.path.join(ROOT, c))):
            pixels = torch.from_numpy(np.array(Image.open(os.path.join(ROOT, c, f)).convert("L")))
            items.append((pixels, label))
    return items


class Mapped(Dataset):
    def __init__(self, items, fn):
        self.items, self.fn = items, fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pixels, label = self.items[i]
        return self.fn(pixels), label   # applied on every access, like a Burn Mapper


def mirror(x):
    return torch.flip(x, dims=[-1])


def build():
    base = load_all()
    augmented = ConcatDataset([Mapped(base, lambda x: x), Mapped(base, mirror)])
    original, label0 = augmented[0]
    mirrored, label1 = augmented[len(base)]
    bright = int(original.int().sum(dim=1).argmax())
    round_trip = torch.equal(mirror(mirrored), original) and label0 == label1
    return (len(base), len(augmented), original[bright].tolist(), mirrored[bright].tolist(),
            round_trip)


def main():
    n, n_aug, before, after, round_trip = build()
    print(f"dataset: {n} images -> with mirrored copies: {n_aug}")
    print("image 0, bright row          =", before)
    print(f"image {n}, same row, mirrored =", after)
    print("mirror twice == original, label kept:", round_trip)


def test_matches_burn():
    n, n_aug, before, after, round_trip = build()
    assert (n, n_aug) == (24, 48)
    assert after == before[::-1]
    assert before == EXPECTED_BEFORE
    assert round_trip


EXPECTED_BEFORE = [24, 32, 3, 18, 22, 231, 229, 239, 223, 251, 245, 17]


if __name__ == "__main__":
    main()
