"""Example 18.2 - training the U-Net by flow matching (PyTorch reference + parity test).

PyTorch has no Learner: this is the loop Burn's Learner runs, written out.
Same data stream, same fixed starting weights, same AdamW as rust/src/main.rs.

Run:   python main.py
Test:  pytest main.py
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def flow_item(seed, index, p_drop):
    rng = Lcg(seed ^ ((index * 2654435761) & 0xFFFFFFFF))
    cls = index % 4
    x1 = to_signed(make_image(cls, rng))
    x0 = rng.normals(PIXELS)
    t = rng.uniform()
    drop = rng.below(100) < p_drop
    return x1, x0, t, NULL_CLASS if drop else cls


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


def det(n, fan_in, phase):
    return torch.tensor([math.sin(i * 2.3 + phase) / math.sqrt(fan_in) for i in range(n)],
                        dtype=torch.float32)


class Fixed:
    def __init__(self):
        self.k = 0

    def phase(self):
        self.k += 1
        return 0.1 * self.k

    def conv(self, m):
        o, i, kh, kw = m.weight.shape
        m.weight.copy_(det(o * i * kh * kw, i * kh * kw, self.phase()).reshape(m.weight.shape))
        m.bias.zero_()

    def conv_t(self, m):
        i, o, kh, kw = m.weight.shape
        m.weight.copy_(det(i * o * kh * kw, i, self.phase()).reshape(m.weight.shape))
        m.bias.zero_()

    def linear(self, m):
        o, i = m.weight.shape
        m.weight.copy_(det(o * i, i, self.phase()).reshape(o, i))
        m.bias.zero_()

    def block(self, b):
        self.conv(b.conv1)
        self.linear(b.cond)
        self.conv(b.conv2)
        if b.skip is not None:
            self.conv(b.skip)


@torch.no_grad()
def with_fixed_weights(model):
    f = Fixed()
    f.linear(model.time1)
    f.linear(model.time2)
    n, d = model.class_emb.weight.shape
    model.class_emb.weight.copy_(det(n * d, 1, f.phase()).reshape(n, d))
    f.conv(model.conv_in)
    f.block(model.enc)
    f.conv(model.down)
    f.block(model.mid)
    f.conv_t(model.up)
    f.block(model.dec)
    f.conv(model.conv_out)
    return model



STEPS, BATCH, LR, P_DROP, SEED = 1200, 64, 3e-3, 10, 18


def batch(seed, start, n, p_drop):
    items = [flow_item(seed, i, p_drop) for i in range(start, start + n)]
    x1 = torch.tensor(np.stack([it[0] for it in items])).reshape(n, 1, SIZE, SIZE)
    x0 = torch.tensor(np.stack([it[1] for it in items])).reshape(n, 1, SIZE, SIZE)
    t = torch.tensor([it[2] for it in items], dtype=torch.float32)
    labels = torch.tensor([it[3] for it in items])
    return x1, x0, t, labels


def flow_loss(model, x1, x0, t, labels):
    tt = t[:, None, None, None]
    xt = x0 * (1.0 - tt) + x1 * tt          # the straight path
    v = model(xt, t, labels)
    return ((v - (x1 - x0)) ** 2).mean()    # target v = x1 - x0


def train(steps=STEPS):
    torch.set_num_threads(1)  # one thread: the fastest on a busy 2-core machine
    model = with_fixed_weights(UNet(16, 32, 64))
    optim = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=1e-4)
    losses = []
    for s in range(steps):
        loss = flow_loss(model, *batch(SEED, s * BATCH, BATCH, P_DROP))
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    with torch.no_grad():
        valid = [flow_loss(model, *batch(7, k * BATCH, BATCH, P_DROP)).item() for k in range(8)]
    return model, losses, valid


def main():
    start = time.time()
    model, losses, valid = train()
    secs = time.time() - start
    print("step range    mean train loss")
    for k in range(0, len(losses), 200):
        chunk = losses[k:k + 200]
        print(f"{k + 1:>4}-{k + len(chunk):<4}    {sum(chunk) / len(chunk):.4f}")
    print(f"validation loss (8 batches, unseen stream) = {sum(valid) / len(valid):.4f}")
    print(f"{len(losses)} steps of N = 64 in {secs:.1f} s (data generation included)")
    print("first 5 losses:", ", ".join(f"{x:.6f}" for x in losses[:5]))


def test_matches_burn():
    _, losses, _ = train(5)
    assert np.allclose(losses, EXPECTED_FIRST_LOSSES, atol=1e-3)


# The first five batch losses of Burn's Learner (train/epoch-1/Loss.log of rust/src/main.rs).
EXPECTED_FIRST_LOSSES = [2.915223, 3.726287, 2.406771, 1.856128, 1.609260]


if __name__ == "__main__":
    main()
