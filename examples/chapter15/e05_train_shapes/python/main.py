"""Example 15.5 - Training the CNN (PyTorch reference + parity test).

Same 160 training images, same order, same fixed initial weights, same Adam
(lr 0.02, eps 1e-8) as rust.rs. The mean loss per epoch must match.

Run:   python main.py
Test:  pytest main.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


class Shapes(Dataset):
    def __init__(self, per_class, seed):
        rng = Lcg(seed)
        self.items = [(make_image(i % 4, rng), i % 4) for i in range(per_class * 4)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pixels, label = self.items[i]
        return torch.from_numpy(pixels).reshape(1, SIZE, SIZE), label


def build():
    train, test = Shapes(40, 42), Shapes(20, 7)
    loader = DataLoader(train, batch_size=16, shuffle=False)

    model = SmallCnn().with_fixed_weights()
    optim = torch.optim.Adam(model.parameters(), lr=0.02, eps=1e-8)

    history = []
    for epoch in range(1, 16):
        total, batches = 0.0, 0
        for images, targets in loader:
            loss = F.cross_entropy(model(images), targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            batches += 1
        history.append(total / batches)
        print(f"epoch {epoch:>2}  loss {history[-1]:.4f}")

    images = torch.stack([test[i][0] for i in range(len(test))])
    labels = torch.tensor([test[i][1] for i in range(len(test))])
    correct = (model(images).argmax(1) == labels).sum().item()
    return history, correct


EXPECTED = [
    1.344356, 1.104616, 0.853722, 0.715782, 0.585857, 0.470269, 0.384723, 0.276252,
    0.130639, 0.056581, 0.031400, 0.028396, 0.021925, 0.020555, 0.009121,
]


def main():
    _, correct = build()
    print(f"test accuracy = {correct}/80")


def test_matches_burn():
    history, correct = build()
    assert all(abs(a - b) < 2e-3 for a, b in zip(history, EXPECTED))
    assert correct == 80


if __name__ == "__main__":
    main()
