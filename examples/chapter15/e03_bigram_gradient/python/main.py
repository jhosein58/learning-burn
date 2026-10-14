"""Example 15.3 - The Bigram as a Trainable Table (PyTorch reference + parity test).

A table of logits W [V, V], softmax and cross-entropy. The gradient on the rows
a batch uses is (p - one_hot) / N; over the whole text it is
(n_a softmax(W[a]) - C[a]) / M with the pair counts C. Both are checked
against autograd; then plain gradient descent with the count form.

Run:   python main.py
Test:  pytest main.py
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TEXT = (Path(__file__).resolve().parents[2] / "data" / "shakespeare.txt").read_bytes()
V, BATCH, LR, STEPS = 62, 8, 100.0, 2000


class Lcg:
    """Tiny LCG, bit-identical to the Rust one."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def uniform(self):
        return self.next() / 32768.0

    def below(self, n):
        hi = self.next()
        lo = self.next()
        return ((hi << 15) | lo) % n


def count_pairs(ids):
    c = torch.zeros(V, V)
    c.index_put_((torch.tensor(ids[:-1]), torch.tensor(ids[1:])),
                 torch.ones(len(ids) - 1), accumulate=True)
    return c


def count_loss(w, c):
    return -(c * torch.log_softmax(w, dim=1)).sum() / c.sum()


def count_grad(w, c):
    return (torch.softmax(w, dim=1) * c.sum(dim=1, keepdim=True) - c) / c.sum()


def batch_grad(w, x, y):
    g = torch.zeros(V, V)
    p = torch.softmax(w, dim=1)
    for a, b in zip(x, y):
        g[a] += (p[a] - F.one_hot(torch.tensor(b), V)) / len(x)
    return g


def build(steps):
    voc = sorted(set(TEXT))
    index = {b: i for i, b in enumerate(voc)}
    ids = [index[b] for b in TEXT]
    n_train = len(ids) * 9 // 10
    c_train, c_val = count_pairs(ids[:n_train]), count_pairs(ids[n_train:])

    rng = Lcg(42)
    u = np.array([rng.uniform() for _ in range(V * V)], dtype=np.float32)
    w0 = torch.tensor(((np.float32(2.0) * u - np.float32(1.0)) * np.float32(0.1)).reshape(V, V))
    w = w0.clone().requires_grad_()

    rng = Lcg(7)
    starts = [rng.below(n_train - 1) for _ in range(BATCH)]
    x = [ids[s] for s in starts]
    y = [ids[s + 1] for s in starts]
    show = lambda i: {32: "_", 10: "/"}.get(voc[i], chr(voc[i]))
    pairs = " ".join(show(a) + show(b) for a, b in zip(x, y))
    loss = F.cross_entropy(w[torch.tensor(x)], torch.tensor(y))
    loss.backward()
    hand = batch_grad(w0, x, y)
    batch_diff = (hand - w.grad).abs().max().item()
    p0 = torch.softmax(w0, dim=1)

    w.grad = None
    count_loss(w, c_train).backward()
    count_diff = (count_grad(w0, c_train) - w.grad).abs().max().item()

    wt = w0.clone()
    log = []
    for step in range(steps + 1):
        if step % 500 == 0:
            log.append((step, count_loss(wt, c_train).item(), count_loss(wt, c_val).item()))
        if step < steps:
            wt = wt - LR * count_grad(wt, c_train)
    p = torch.softmax(wt, dim=1)
    counts = c_train / c_train.sum(dim=1, keepdim=True)
    table = [(p[index[a], index[b]].item(), counts[index[a], index[b]].item())
             for a, b in [(ord("T"), ord("h")), (ord("q"), ord("u"))]]
    return dict(pairs=pairs, loss=loss.item(), target_grad=hand[x[0], y[0]].item(),
                target_p=p0[x[0], y[0]].item(), batch_diff=batch_diff,
                count_diff=count_diff, log=log, table=table)


def main():
    r = build(STEPS)
    print(f"batch of {BATCH} pairs: {r['pairs']}")
    print(f"batch loss = {r['loss']:.6f}")
    print(f"hand dL/dW[x0][y0] = {r['target_grad']:.6f}  with p(y0 | x0) = {r['target_p']:.6f}")
    print(f"batch form: max|hand - autograd| = {r['batch_diff']:.1e}")
    print(f"count form: max|hand - autograd| = {r['count_diff']:.1e}")
    for step, tr, va in r["log"]:
        print(f"step {step:>4}  train {tr:.4f}  val {va:.4f}  val perplexity {math.exp(va):.2f}")
    (th, thc), (qu, quc) = r["table"]
    print(f"p(h | T) = {th:.3f} (counts {thc:.3f}),  p(u | q) = {qu:.3f} (counts {quc:.3f})")


def test_matches_burn():
    r = build(500)
    assert r["batch_diff"] < 1e-7 and r["count_diff"] < 1e-7
    assert abs(r["loss"] - 4.107936) < 1e-5
    assert abs(r["target_grad"] - (-0.120597)) < 1e-6
    burn = [(4.1233, 4.1210), (2.4121, 2.5055)]
    assert all(abs(tr - a) < 1e-4 and abs(va - b) < 1e-4
               for (_, tr, va), (a, b) in zip(r["log"], burn))


if __name__ == "__main__":
    main()
