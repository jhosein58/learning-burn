"""Example 23.4 - A GRU, one frame at a time (PyTorch reference + test).

nn.GRU with weights copied from the same LCG as rust/src/main.rs, run over the
whole sequence and then one frame per call with the hidden state passed back
in; and the step written out from PyTorch's equations.

Run:   python main.py
Test:  pytest main.py
"""

import torch
import torch.nn as nn

I, H, T = 3, 4, 10


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.s >> 16) & 0x7FFF

    def grid(self, n, m, div):
        return [((self.next() % (2 * m + 1)) - m) / div for _ in range(n)]


def gru_step(x, h, w_ih, w_hh, b_ih, b_hh):
    a = w_ih @ x + b_ih                    # [3H]: rows r, z, n
    b = w_hh @ h + b_hh
    r = torch.sigmoid(a[:H] + b[:H])
    z = torch.sigmoid(a[H:2 * H] + b[H:2 * H])
    n = torch.tanh(a[2 * H:] + r * b[2 * H:])
    return (1 - z) * n + z * h


@torch.no_grad()
def build():
    rng = Lcg(2005)
    w_ih = torch.tensor(rng.grid(3 * H * I, 16, 32.0)).reshape(3 * H, I)
    w_hh = torch.tensor(rng.grid(3 * H * H, 16, 32.0)).reshape(3 * H, H)
    b_ih = torch.tensor(rng.grid(3 * H, 16, 32.0))
    b_hh = torch.tensor(rng.grid(3 * H, 16, 32.0))
    x = torch.tensor(rng.grid(T * I, 8, 4.0)).reshape(1, T, I)

    gru = nn.GRU(I, H, batch_first=True)
    gru.weight_ih_l0.copy_(w_ih)
    gru.weight_hh_l0.copy_(w_hh)
    gru.bias_ih_l0.copy_(b_ih)
    gru.bias_hh_l0.copy_(b_hh)

    whole, _ = gru(x)                              # [1, T, H]
    h, steps = torch.zeros(1, 1, H), []
    for t in range(T):
        y, h = gru(x[:, t:t + 1], h)               # one frame, state in and out
        steps.append(y)
    steps = torch.cat(steps, dim=1)

    hh, by_hand = torch.zeros(H), []
    for t in range(T):
        hh = gru_step(x[0, t], hh, w_ih, w_hh, b_ih, b_hh)
        by_hand.append(hh)
    by_hand = torch.stack(by_hand).unsqueeze(0)
    return whole, steps, by_hand


def main():
    whole, steps, by_hand = build()
    print(f"h[{T - 1}] nn.GRU, whole = {[round(v, 6) for v in whole[0, -1].tolist()]}")
    print(f"h[{T - 1}] nn.GRU, steps = {[round(v, 6) for v in steps[0, -1].tolist()]}")
    print(f"h[{T - 1}] by hand       = {[round(v, 6) for v in by_hand[0, -1].tolist()]}")
    print(f"max|whole - steps| = {(whole - steps).abs().max().item():.3g}")
    print(f"max|whole - hand|  = {(whole - by_hand).abs().max().item():.3g}")


def test_matches_burn():
    whole, steps, by_hand = build()
    assert (whole - steps).abs().max() < 1e-6
    assert (whole - by_hand).abs().max() < 1e-6
    # rust/src/main.rs: h[9] by hand and from Burn's Gru.
    expected = torch.tensor([0.149455, 0.421631, 0.50154, -0.536617])
    assert (whole[0, -1] - expected).abs().max() < 1e-5


if __name__ == "__main__":
    main()
