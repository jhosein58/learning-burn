"""Example 23.5 - Train on whole clips, run one frame at a time (PyTorch twin).

The burst detector of rust/src/main.rs: three dilated causal convs (F.pad on
the left), nn.GRU and a linear head, with the same LCG weights and the same
LCG data, trained with Adam in the same batch order. Then the same weights
run one frame per call, the conv buffers and the GRU state carried outside.

Run:   python main.py
Test:  pytest main.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T, BATCH, TRAIN, TEST = 96, 32, 256, 64
K, DILATIONS, C, H = 3, [1, 2, 4], 8, 16


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.s >> 16) & 0x7FFF

    def below(self, n):
        return self.next() % n

    def grid(self, n, m, div):
        return [((self.next() % (2 * m + 1)) - m) / div for _ in range(n)]


def clip(rng):
    length = 16 + rng.below(17)
    start = 8 + rng.below(T - 8 - length)
    x = np.array(rng.grid(T, 32, 64.0), dtype=np.float32)
    y = np.zeros(T, dtype=np.float32)
    for t in range(start, start + length):
        x[t] += np.float32(math.sin(2 * math.pi * (t - start) / 8))   # f32 add, as in Rust
        y[t] = 1
    return x, y


def dataset(n, seed):
    rng = Lcg(seed)
    xs, ys = zip(*(clip(rng) for _ in range(n)))
    return torch.tensor(np.stack(xs)).unsqueeze(1), torch.tensor(np.stack(ys))


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        rng = Lcg(7)
        self.convs = nn.ModuleList()
        for i, d in enumerate(DILATIONS):
            c_in = 1 if i == 0 else C
            conv = nn.Conv1d(c_in, C, K, dilation=d)
            div = 32.0 if i == 0 else 128.0
            with torch.no_grad():
                conv.weight.copy_(torch.tensor(rng.grid(C * c_in * K, 16, div)).reshape(C, c_in, K))
                conv.bias.zero_()
            self.convs.append(conv)
        self.gru = nn.GRU(C, H, batch_first=True)
        w_ih, w_hh = [], []
        for _ in range(3):  # gates r, z, n; Burn's [in, out] blocks, transposed
            w_ih.append(torch.tensor(rng.grid(C * H, 16, 64.0)).reshape(C, H).T)
            w_hh.append(torch.tensor(rng.grid(H * H, 16, 64.0)).reshape(H, H).T)
        self.head = nn.Linear(H, 1)
        with torch.no_grad():
            self.gru.weight_ih_l0.copy_(torch.cat(w_ih))
            self.gru.weight_hh_l0.copy_(torch.cat(w_hh))
            self.gru.bias_ih_l0.zero_()
            self.gru.bias_hh_l0.zero_()
            self.head.weight.copy_(torch.tensor(rng.grid(H, 16, 64.0)).reshape(1, H))
            self.head.bias.zero_()

    def forward(self, x):                                   # [N, 1, T] -> [N, T]
        h = x
        for conv in self.convs:
            h = F.relu(conv(F.pad(h, (conv.dilation[0] * (K - 1), 0))))
        h, _ = self.gru(h.transpose(1, 2))
        return self.head(h).squeeze(2)

    def zero_state(self, n):
        bufs = [torch.zeros(n, c.in_channels, c.dilation[0] * (K - 1)) for c in self.convs]
        return bufs, torch.zeros(1, n, H)

    def step(self, x_t, state):                             # [N, 1, 1] -> [N, 1]
        bufs, h_state = state
        h, new_bufs = x_t, []
        for conv, buf in zip(self.convs, bufs):
            window = torch.cat([buf, h], dim=2)
            h = F.relu(conv(window))
            new_bufs.append(window[:, :, 1:])
        out, h_state = self.gru(h.transpose(1, 2), h_state)
        return self.head(out[:, 0]), (new_bufs, h_state)


def train(epochs, max_steps=None):
    xs, ys = dataset(TRAIN, 42)
    model = Detector()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for _ in range(epochs):
        for b in range(TRAIN // BATCH):
            x, y = xs[b * BATCH:(b + 1) * BATCH], ys[b * BATCH:(b + 1) * BATCH]
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if len(losses) == max_steps:
                return model, losses
    return model, losses


@torch.no_grad()
def whole_and_streamed(model, x):
    whole = model(x)
    state, frames = model.zero_state(x.shape[0]), []
    for t in range(x.shape[2]):
        logit, state = model.step(x[:, :, t:t + 1], state)
        frames.append(logit)
    return whole, torch.cat(frames, dim=1)


def main():
    torch.set_num_threads(2)
    model, losses = train(12)
    for i, l in enumerate(losses):
        if i == 0 or (i + 1) % 24 == 0:
            print(f"step {i + 1:>3}  loss {l:.4f}")
    print("first five losses:", [round(l, 6) for l in losses[:5]])
    x, y = dataset(TEST, 7)
    whole, streamed = whole_and_streamed(model, x)
    correct = int(((streamed > 0).float() == y).sum())
    print(f"test frames correct = {correct}/{TEST * T} ({100 * correct / (TEST * T):.1f}%)")
    print(f"max|whole - streamed| = {(whole - streamed).abs().max().item():.1e}")


def test_matches_burn():
    model, losses = train(1, 5)
    # rust/src/main.rs: the first five Adam steps.
    expected = [0.692895, 0.680786, 0.665949, 0.645024, 0.623308]
    assert max(abs(a - b) for a, b in zip(losses, expected)) < 1e-4
    x, _ = dataset(8, 7)
    whole, streamed = whole_and_streamed(model, x)
    assert (whole - streamed).abs().max() < 1e-5


if __name__ == "__main__":
    main()
