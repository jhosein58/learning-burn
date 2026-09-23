"""Example 17.2 - A Dataset over a Folder of Images (PyTorch reference + parity test).

One sub-folder per class, one PNG per example. The Dataset lists files up
front and decodes an image only when it is asked for. Same labels, same bytes,
same batch as rust.rs.

Run:   python main.py
Test:  pytest main.py
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "shapes")


class ImageFolder(Dataset):
    def __init__(self, root):
        self.classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.items = [(os.path.join(root, c, f), label)
                      for label, c in enumerate(self.classes)
                      for f in sorted(os.listdir(os.path.join(root, c))) if f.endswith(".png")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        pixels = np.array(Image.open(path).convert("L"))   # decode happens HERE
        return torch.from_numpy(pixels).float().div(255).unsqueeze(0), label


def build():
    data = ImageFolder(ROOT)
    path, label = data.items[7]
    pixel_sum = int(np.array(Image.open(path).convert("L"), dtype=np.uint32).sum())
    images, labels = next(iter(DataLoader(data, batch_size=8, shuffle=False)))
    return (data.classes, len(data), label, pixel_sum, list(images.shape), labels.tolist(),
            images.mean().item())


def main():
    classes, n, label, pixel_sum, dims, labels, mean = build()
    print("classes     =", classes)
    print("images      =", n)
    print(f"item 7      = label {label}, pixel sum {pixel_sum}")
    print(f"first batch = {dims}, labels {labels}, mean pixel {mean:.4f}")


def test_matches_burn():
    classes, n, label, pixel_sum, dims, labels, mean = build()
    assert classes == ["0_horizontal", "1_vertical", "2_diagonal", "3_plus"]
    assert (n, label) == (24, 1)
    assert pixel_sum == EXPECTED_SUM
    assert dims == [8, 1, 12, 12]
    assert labels == [0, 0, 0, 0, 0, 0, 1, 1]
    assert abs(mean - EXPECTED_MEAN) < 1e-5


EXPECTED_SUM = 4017
EXPECTED_MEAN = 0.110069


if __name__ == "__main__":
    main()
