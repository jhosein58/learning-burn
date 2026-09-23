"""Example 15.6 - What the Filters Learned, and an int8 CNN (PyTorch reference + parity test).

Trains exactly as Example 15.5, prints the four 3x3 filters of the first
layer, then quantises every weight tensor to int8 (Chapter 14's scheme) and
checks the accuracy survives.

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


def train():
    model = SmallCnn().with_fixed_weights()
    optim = torch.optim.Adam(model.parameters(), lr=0.02, eps=1e-8)
    loader = DataLoader(Shapes(40, 42), batch_size=16, shuffle=False)
    for _ in range(15):
        for images, targets in loader:
            loss = F.cross_entropy(model(images), targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model


def int8_round_trip(w):
    scale = w.abs().max() / 127
    codes = torch.round(w / scale).clamp(-127, 127)   # whole numbers, one byte each
    return codes * scale, scale.item()


def build():
    model = train()
    torch.set_grad_enabled(False)   # inference only from here on
    test = Shapes(20, 7)
    images = torch.stack([test[i][0] for i in range(len(test))])
    labels = torch.tensor([test[i][1] for i in range(len(test))])

    filters = model.conv1.weight.flatten().tolist()
    f32_correct = (model(images).argmax(1) == labels).sum().item()

    scales = []
    for layer in (model.conv1, model.conv2, model.head):
        w, s = int8_round_trip(layer.weight)
        layer.weight.copy_(w)
        scales.append(s)
    print("scales         = [" + ", ".join(f"{s:.5f}" for s in scales) + "]")
    int8_correct = (model(images).argmax(1) == labels).sum().item()

    torch.set_grad_enabled(True)
    weights, biases = 36 + 288 + 32, 4 + 8 + 4
    return filters, f32_correct, int8_correct, (weights + biases) * 4, weights + 3 * 4 + biases * 4


EXPECTED = [
    -0.422270, 0.930256, -0.156822, 0.394610, 0.132003, -0.237828,
    1.001078, -0.483036, -0.079344, 0.456176, -0.339555, 0.453012,
    0.761879, -1.099708, 0.817598, 0.453400, -0.468780, 0.602896,
    -0.596102, 0.489929, 0.240630, -1.281593, 0.889172, 0.521594,
    -0.261456, 1.098831, -0.865395, -1.319253, 1.045184, -0.067182,
    0.521430, 0.890422, -1.672505, -0.141084, 1.779191, -1.678253,
]


def main():
    filters, f32_correct, int8_correct, f32_bytes, int8_bytes = build()
    for k in range(4):
        f = filters[k * 9:(k + 1) * 9]
        rows = " ".join("[" + ", ".join(f"{v:.2f}" for v in f[r * 3:r * 3 + 3]) + "]" for r in range(3))
        print(f"conv1 filter {k}: {rows}")
    print(f"f32  model: {f32_correct}/80 correct, {f32_bytes} bytes")
    print(f"int8 model: {int8_correct}/80 correct, {int8_bytes} bytes")


def test_matches_burn():
    filters, f32_correct, int8_correct, f32_bytes, int8_bytes = build()
    assert all(abs(a - b) < 2e-3 for a, b in zip(filters, EXPECTED))
    assert (f32_correct, int8_correct) == (80, 80)
    assert (f32_bytes, int8_bytes) == (1488, 432)


if __name__ == "__main__":
    main()
