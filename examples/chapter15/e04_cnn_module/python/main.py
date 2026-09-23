"""Example 15.4 - A Small CNN as a Module (PyTorch reference + parity test).

The same 12x12 "shapes" data and the same fixed weights as rust.rs, so the
two frameworks must produce the same logits. Also counts parameters against
an MLP that does the same job without convolutions.

Run:   python main.py
Test:  pytest main.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 12


class Lcg:
    """Tiny LCG, bit-identical to the Rust one."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def below(self, n):
        return self.next() % n


def make_image(cls, rng):
    img = np.zeros(SIZE * SIZE, dtype=np.float32)
    if cls == 0:
        r, c = rng.below(12), rng.below(7)
        for j in range(6):
            img[r * SIZE + c + j] = 1.0
    elif cls == 1:
        c, r = rng.below(12), rng.below(7)
        for i in range(6):
            img[(r + i) * SIZE + c] = 1.0
    elif cls == 2:
        r, c = rng.below(7), rng.below(7)
        for i in range(6):
            img[(r + i) * SIZE + c + i] = 1.0
    else:
        r, c = 2 + rng.below(8), 2 + rng.below(8)
        for d in range(5):
            img[r * SIZE + c + d - 2] = 1.0
            img[(r + d - 2) * SIZE + c] = 1.0
    for p in range(SIZE * SIZE):
        img[p] += np.float32(rng.next()) / np.float32(32768.0) * np.float32(0.25)
    return img


def det(n, fan_in, phase):
    return [math.sin(i * 2.3 + phase) / math.sqrt(fan_in) for i in range(n)]


class SmallCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.gap(x).flatten(1)
        return self.head(x)

    def with_fixed_weights(self):
        with torch.no_grad():
            self.conv1.weight.copy_(torch.tensor(det(36, 9, 0.1)).reshape(4, 1, 3, 3))
            self.conv1.bias.zero_()
            self.conv2.weight.copy_(torch.tensor(det(288, 36, 0.3)).reshape(8, 4, 3, 3))
            self.conv2.bias.zero_()
            self.head.weight.copy_(torch.tensor(det(32, 8, 0.5)).reshape(4, 8))
            self.head.bias.zero_()
        return self


def build():
    rng = Lcg(42)
    images = np.stack([make_image(c, rng) for c in range(4)])
    x = torch.from_numpy(images).reshape(4, 1, SIZE, SIZE)
    model = SmallCnn().with_fixed_weights()
    logits = model(x)
    print("logits shape =", list(logits.shape))
    cnn = sum(p.numel() for p in model.parameters())
    mlp = nn.Sequential(nn.Linear(SIZE * SIZE, 32), nn.ReLU(), nn.Linear(32, 4))
    return logits.detach().flatten().tolist(), cnn, sum(p.numel() for p in mlp.parameters())


EXPECTED = [
    0.000385, 0.000009, -0.000369, -0.000674, 0.007906, 0.007646, 0.005866, 0.002920,
    -0.001834, -0.001361, -0.000618, 0.000247, -0.001481, -0.000899, -0.000138, 0.000650,
]


def main():
    logits, cnn, mlp = build()
    for c in range(4):
        row = ", ".join(f"{v:.4f}" for v in logits[c * 4:c * 4 + 4])
        print(f"image of class {c}: logits [{row}]")
    print("CNN parameters =", cnn)
    print("MLP parameters =", mlp, " (144 -> 32 -> 4)")


def test_matches_burn():
    logits, cnn, mlp = build()
    assert all(abs(a - b) < 1e-5 for a, b in zip(logits, EXPECTED))
    assert cnn == 372
    assert mlp == 4772


if __name__ == "__main__":
    main()
