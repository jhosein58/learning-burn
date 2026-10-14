"""Example 15.6 - Greedy, Temperature, Top-k and Top-p by Hand (PyTorch reference + parity test).

Each decoding strategy turns one logits vector into a probability vector;
a token is then drawn by the inverse CDF at a uniform u. The uniforms come
from the same LCG as the Rust example, not from torch's RNG, so the draws
and their counts must match exactly.

Run:   python main.py
Test:  pytest main.py
"""

import math

import torch

Z = [2.0, 1.0, 0.5, 0.0, -1.0]
TOKENS = ["a", "b", "c", "d", "e"]
DRAWS = 10_000


class Lcg:
    """Tiny LCG, bit-identical to the Rust one."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def uniform(self):
        return self.next() / 32768.0


def greedy(z):
    return int(z.argmax())


def with_temperature(z, temp):
    return torch.softmax(z / temp, dim=0)


def top_k(z, k, temp):
    kth = z.sort(descending=True).values[k - 1]
    return with_temperature(z.masked_fill(z < kth, -1e9), temp)


def top_p(z, p, temp):
    probs = with_temperature(z, temp)
    sorted_p = probs.sort(descending=True).values
    before = sorted_p.cumsum(0) - sorted_p
    kept = int((before < p).sum())
    probs = probs.masked_fill(probs < sorted_p[kept - 1], 0.0)
    return probs / probs.sum()


def sample(probs, u):
    cdf = torch.tensor(0.0)
    for i, q in enumerate(probs):
        cdf = cdf + q          # accumulate in float32, as the Rust loop does
        if u < cdf:
            return i
    return len(probs) - 1


def entropy(p):
    return sum(-q * math.log(q) for q in p if q > 0)


def build():
    z = lambda: torch.tensor(Z)
    choices = [("T = 1", with_temperature(z(), 1.0)), ("T = 0.5", with_temperature(z(), 0.5)),
               ("T = 2", with_temperature(z(), 2.0)), ("top-k 2", top_k(z(), 2, 1.0)),
               ("top-p 0.8", top_p(z(), 0.8, 1.0))]
    strategies = []
    for name, p in choices:
        rng = Lcg(5)
        us = [rng.uniform() for _ in range(DRAWS)]
        draws = [sample(p, u) for u in us]
        counts = [draws.count(i) for i in range(5)]
        strategies.append((name, p.tolist(), draws[:10], counts))
    return greedy(z()), us[:10], strategies


def main():
    g, us, strategies = build()
    print(f"logits z = {Z}  tokens {TOKENS}".replace("'", '"'))
    print(f"greedy -> {TOKENS[g]}")
    print("first 10 uniforms u = " + " ".join(f"{u:.3f}" for u in us))
    print("strategy     p(a)   p(b)   p(c)   p(d)   p(e)   entropy")
    for name, p, _, _ in strategies:
        print(f"{name:<10}  " + " ".join(f"{q:.4f}" for q in p) + f"  {entropy(p):.3f}")
    print(f"strategy    first 10 draws        counts in {DRAWS} draws")
    for name, _, draws, counts in strategies:
        print(f"{name:<10}  " + " ".join(TOKENS[i] for i in draws) + f"   {counts}")


def test_matches_burn():
    g, _, strategies = build()
    assert g == 0
    burn_counts = [[5583, 2071, 1260, 781, 305], [8258, 1132, 430, 156, 24],
                   [3701, 2281, 1765, 1376, 877], [7267, 2733, 0, 0, 0],
                   [6220, 2346, 1434, 0, 0]]
    assert [s[3] for s in strategies] == burn_counts
    assert abs(strategies[0][1][0] - 0.5630) < 1e-4


if __name__ == "__main__":
    main()
