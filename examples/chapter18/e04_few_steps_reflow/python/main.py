"""Example 18.4 - fewer steps, and one round of reflow (PyTorch reference + parity test).

Starts from the weights Burn trained in Example 18.2 (weights/flow_unet.safetensors),
sweeps the number of Euler steps, then makes 1024 (noise, sample) pairs with the
16-step guided sampler and fine-tunes on straight paths between them, as
rust/src/main.rs does with the Learner.

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

W, TEACHER_STEPS = 2.0, 16


def make_pairs(model, m, seed):
    labels = [i % 4 for i in range(m)]
    x0 = noise(m, seed)
    x1 = sample(model, x0, labels, W, TEACHER_STEPS)
    return x0, x1, torch.tensor(labels)


def reflow(model, pairs, steps):
    """Fine-tune on straight paths between the two ends of each pair; item i is pair i mod m."""
    x0_all, x1_all, y_all = pairs
    m = len(y_all)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=1e-4)
    model.train()
    losses = []
    for s in range(steps):
        idx = list(range(s * 64, (s + 1) * 64))
        rows = [i % m for i in idx]
        t = torch.tensor([Lcg(31 ^ ((i * 2654435761) & 0xFFFFFFFF)).uniform() for i in idx])
        x0, x1, y = x0_all[rows], x1_all[rows], y_all[rows]
        tt = t[:, None, None, None]
        v = model(x0 * (1.0 - tt) + x1 * tt, t, y)
        loss = ((v - (x1 - x0)) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return model.eval(), losses


def main():
    torch.set_num_threads(1)
    base = trained()
    tpl = templates()
    labels = [i % 4 for i in range(256)]
    z = noise(256, 5)
    teacher = sample(base, z, labels, W, TEACHER_STEPS)
    rows = []

    def evaluate(name, model, w, steps):
        x = sample(model, z, labels, w, steps)
        hits, to_shape = score(x.numpy(), labels, tpl)
        rows.append((name, w, steps, hits, to_shape, ((x - teacher) ** 2).mean().item()))

    for steps in [1, 2, 4, 8, 16, 32]:
        evaluate("base", base, W, steps)
    student, losses = reflow(trained(), make_pairs(base, 1024, 11), 400)
    for steps in [1, 2, 4]:
        evaluate("reflow", student, 1.0, steps)

    print("model    w  steps  right shape  to nearest shape  to 16-step teacher")
    for name, w, steps, hits, to_shape, to_teacher in rows:
        print(f"{name:<6} {w:>3.1f} {steps:>6} {hits:>11.1f}% {to_shape:>17.4f} {to_teacher:>19.4f}")
    print(f"reflow loss: first 50 steps {np.mean(losses[:50]):.4f}, "
          f"last 50 steps {np.mean(losses[-50:]):.4f}")


def small_run():
    pairs = make_pairs(trained(), 64, 11)
    _, losses = reflow(trained(), pairs, 3)
    return pairs[1][3].flatten()[:4].tolist(), losses


def test_matches_burn():
    pixels, losses = small_run()
    assert np.allclose(pixels, EXPECTED_PAIR, atol=1e-3)
    assert np.allclose(losses, EXPECTED_LOSSES, atol=1e-3)


# Pair 3's first four target pixels and the first three reflow losses (the Rust test uses the same).
EXPECTED_PAIR = [-0.793831, -0.681093, -0.665306, -0.655993]
EXPECTED_LOSSES = [0.094534, 0.085923, 0.086509]


if __name__ == "__main__":
    import sys

    if "--expected" in sys.argv:
        print(small_run())
    else:
        main()
