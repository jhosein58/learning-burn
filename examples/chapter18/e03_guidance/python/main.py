"""Example 18.3 - classifier-free guidance, swept (PyTorch reference + parity test).

Loads the weights Burn trained in Example 18.2 (weights/flow_unet.safetensors,
written by burn-store with PyTorch names and layout) and samples exactly as
rust/src/main.rs does: the same noise, the same Euler steps, the same guidance.

Run:   python main.py
Test:  pytest main.py
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

SIZE = 12
PIXELS = SIZE * SIZE
NULL_CLASS = 4


class Lcg:
    """Tiny LCG, bit-identical to the Rust one."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def below(self, n):
        return self.next() % n

    def uniform(self):
        return np.float32((np.float32(self.next()) + np.float32(0.5)) / np.float32(32768.0))

    def normals(self, n):
        out = []
        while len(out) < n:
            u1 = (self.next() + 1.0) / 32768.0
            u2 = self.next() / 32768.0
            r = math.sqrt(-2.0 * math.log(u1))
            a = 2.0 * math.pi * u2
            out += [r * math.cos(a), r * math.sin(a)]
        return np.array(out[:n], dtype=np.float32)


def make_image(cls, rng):
    img = np.zeros(PIXELS, dtype=np.float32)
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
    for p in range(PIXELS):
        img[p] += np.float32(rng.next()) / np.float32(32768.0) * np.float32(0.25)
    return img


def to_signed(img):
    return (np.float32(1.6) * img - np.float32(1.0)).astype(np.float32)


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.tensor([math.exp(-math.log(10000.0) * i / half) for i in range(half)],
                         dtype=torch.float32)
    args = (t[:, None] * 1000.0) * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, d_cond, groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.cond = nn.Linear(d_cond, c_out)
        self.norm2 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x, c):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond(F.silu(c))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + (x if self.skip is None else self.skip(x))


class UNet(nn.Module):
    def __init__(self, c=16, t_dim=32, d=64):
        super().__init__()
        g = 4
        self.t_dim = t_dim
        self.time1 = nn.Linear(t_dim, d)
        self.time2 = nn.Linear(d, d)
        self.class_emb = nn.Embedding(5, d)
        self.conv_in = nn.Conv2d(1, c, 3, padding=1)
        self.enc = ResBlock(c, c, d, g)
        self.down = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)
        self.mid = ResBlock(2 * c, 2 * c, d, g)
        self.up = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec = ResBlock(2 * c, c, d, g)
        self.norm_out = nn.GroupNorm(g, c)
        self.conv_out = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, x, t, labels):
        c = self.time2(F.silu(self.time1(time_embedding(t, self.t_dim)))) + self.class_emb(labels)
        h = self.conv_in(x)
        skip = self.enc(h, c)
        h = self.mid(self.down(skip), c)
        h = torch.cat([self.up(h), skip], dim=1)
        h = self.dec(h, c)
        return self.conv_out(F.silu(self.norm_out(h)))



@torch.no_grad()
def sample(model, noise, labels, w, steps):
    """Euler steps along the learned velocity, with classifier-free guidance."""
    n = len(labels)
    y = torch.cat([torch.tensor(labels), torch.full((n,), NULL_CLASS)])
    dt = 1.0 / steps
    x = noise
    for k in range(steps):
        t = torch.full((2 * n,), k * dt)
        v = model(torch.cat([x, x]), t, y)
        v_c, v_u = v[:n], v[n:]
        x = x + (v_u + w * (v_c - v_u)) * dt
    return x


def templates():
    """Every clean shape at every position: 0.8 on the shape, -0.8 elsewhere."""
    out = []

    def add(cls, cells):
        img = np.full(PIXELS, -0.8, dtype=np.float32)
        img[cells] = 0.8
        out.append((cls, img))

    for r in range(12):
        for c in range(7):
            add(0, [r * SIZE + c + j for j in range(6)])
    for c in range(12):
        for r in range(7):
            add(1, [(r + i) * SIZE + c for i in range(6)])
    for r in range(7):
        for c in range(7):
            add(2, [(r + i) * SIZE + c + i for i in range(6)])
    for r in range(2, 10):
        for c in range(2, 10):
            add(3, [p for d in range(5) for p in (r * SIZE + c + d - 2, (r + d - 2) * SIZE + c)])
    return np.array([c for c, _ in out]), np.stack([t for _, t in out])


def score(images, labels, tpl):
    """(% of samples whose nearest shape is the requested class, mean distance to it)."""
    classes, t = tpl
    x = images.reshape(len(images), PIXELS)
    d = ((x[:, None, :] - t[None]) ** 2).mean(-1)
    nearest = d.argmin(1)
    hits = (classes[nearest] == np.array(labels)).mean() * 100.0
    return hits, d.min(1).mean()


def noise(n, seed):
    return torch.tensor(Lcg(seed).normals(n * PIXELS)).reshape(n, 1, SIZE, SIZE)


WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "weights")


def trained():
    """Example 18.2's model, from the safetensors file Burn wrote (PyTorch names and layout)."""
    model = UNet(16, 32, 64)
    model.load_state_dict(load_file(os.path.join(WEIGHTS, "flow_unet.safetensors")))
    return model.eval()

SWEEP = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
STEPS = 16
PER_CLASS = 64


def build():
    torch.set_num_threads(1)
    model = trained()
    labels = [i % 4 for i in range(4 * PER_CLASS)]
    z = noise(len(labels), 5)
    tpl = templates()
    rows = []
    for w in SWEEP:
        x = sample(model, z, labels, w, STEPS).numpy()
        rows.append((w, *score(x, labels, tpl)))
    return rows


def data_score():
    rng = Lcg(7)
    labels = [i % 4 for i in range(4 * PER_CLASS)]
    x = np.stack([to_signed(make_image(c, rng)) for c in labels])
    return score(x, labels, templates())


def main():
    print(f"{4 * PER_CLASS} samples ({PER_CLASS} per class), {STEPS} Euler steps")
    print("guidance w   right shape   distance to nearest shape")
    for w, hits, dist in build():
        print(f"{w:>10.1f}   {hits:>10.1f}%   {dist:>10.4f}")
    hits, dist = data_score()
    print(f"real data    {hits:>10.1f}%   {dist:>10.4f}")


def four_samples():
    return sample(trained(), noise(4, 99), [0, 1, 2, 3], 2.0, 8).numpy().reshape(4, PIXELS)


def print_expected():
    x = four_samples()
    for i in range(4):
        print("[" + ", ".join(f"{v:.6f}" for v in x[i, :4]) + "],")
    print(f"EXPECTED_MEAN = {x.mean():.6f}")


def test_matches_burn():
    x = four_samples()
    assert np.allclose(x[:, :4], EXPECTED, atol=1e-3)
    assert abs(x.mean() - EXPECTED_MEAN) < 1e-4


# The same four samples; rust/src/main.rs checks Burn against these numbers.
EXPECTED = [
    [-0.817837, -0.765963, -0.768484, -0.816734],
    [-0.882500, -0.738397, -0.694633, -0.841591],
    [-0.876325, -0.825321, -0.700350, -0.955832],
    [-0.692059, -0.727396, -0.798984, -0.742959],
]
EXPECTED_MEAN = -0.705994


if __name__ == "__main__":
    import sys

    print_expected() if "--expected" in sys.argv else main()
