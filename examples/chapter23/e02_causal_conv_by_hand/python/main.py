"""Example 23.2 - A causal convolution, whole clip and one frame at a time.

PyTorch's Conv1d pads both sides equally, so a causal convolution is F.pad on
the left followed by F.conv1d with no padding. The streaming form keeps the
last d(k-1) input frames in a buffer, here a shift buffer built with torch.cat
(the Rust side uses a ring buffer). Same LCG integers as rust/src/main.rs, so
every number agrees exactly.

Run:   python main.py
Test:  pytest main.py
"""

import torch
import torch.nn.functional as F

T, K = 24, 3
DILATIONS = [1, 2, 4, 8]
CH = [1, 2, 2, 2, 2]


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.s >> 16) & 0x7FFF

    def ints(self, n, lo, hi):
        return [float(lo + self.next() % (hi - lo + 1)) for _ in range(n)]


def layers(rng):
    out = []
    for i, d in enumerate(DILATIONS):
        w = torch.tensor(rng.ints(CH[i + 1] * CH[i] * K, -1, 1)).reshape(CH[i + 1], CH[i], K)
        b = torch.tensor(rng.ints(CH[i + 1], -1, 1))
        out.append((w, b, d))
    return out


def causal_conv(x, w, b, d):
    """x: [N, c_in, T]. Pad d(k-1) zeros on the left only."""
    return F.conv1d(F.pad(x, (d * (K - 1), 0)), w, b, dilation=d)


def stack_whole(x, ls):
    h = x
    for i, (w, b, d) in enumerate(ls):
        h = causal_conv(h, w, b, d)
        if i + 1 < len(ls):
            h = F.relu(h)
    return h


def stack_stream(x, ls):
    """One frame at a time; each layer keeps its last d(k-1) inputs."""
    bufs = [torch.zeros(1, w.shape[1], d * (K - 1)) for w, _, d in ls]
    outs = []
    for t in range(x.shape[2]):
        h = x[:, :, t:t + 1]                              # [1, c_in, 1]
        for i, (w, b, d) in enumerate(ls):
            window = torch.cat([bufs[i], h], dim=2)       # [1, c_in, d(k-1) + 1]
            bufs[i] = window[:, :, 1:]                    # drop the oldest frame
            h = F.conv1d(window, w, b, dilation=d)        # exactly one output
            if i + 1 < len(ls):
                h = F.relu(h)
        outs.append(h)
    return torch.cat(outs, dim=2)


def build():
    rng = Lcg(14)
    x = torch.tensor(rng.ints(T, -2, 2)).reshape(1, 1, T)
    ls = layers(rng)
    w1 = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0, 1.0]).reshape(2, 1, K)
    one = causal_conv(x, w1, None, 2)
    whole = stack_whole(x, ls)
    stream = stack_stream(x, ls)
    x2 = x.clone()
    x2[0, 0, 10] += 5.0
    moved = stack_whole(x2, ls)
    first_moved = int((moved != whole).any(dim=1)[0].nonzero()[0])
    return x, one, whole, stream, first_moved


def main():
    x, one, whole, stream, first_moved = build()
    fmt = lambda v: "".join(f"{a:>3g}" for a in v.tolist())
    print("x           =" + fmt(x[0, 0]))
    print("torch   y[0]=" + fmt(one[0, 0]))
    print("torch   y[1]=" + fmt(one[0, 1]))
    print("whole   y[0]=" + fmt(whole[0, 0]))
    print("stream  y[0]=" + fmt(stream[0, 0]))
    print("max|whole - stream| =", (whole - stream).abs().max().item())
    print(f"x[10] += 5: outputs 0..{first_moved - 1} unchanged, output {first_moved} moves")


def test_matches_burn():
    x, one, whole, stream, first_moved = build()
    # rust/src/main.rs prints the same numbers (loops, Burn's conv1d, ring buffer).
    assert one[0, 0, :8].tolist() == [2.0, 0.0, -1.0, 0.0, -1.0, 1.0, 3.0, 0.0]
    assert whole[0, 0, :8].tolist() == [0.0, 9.0, 6.0, 2.0, 6.0, 7.0, 2.0, 7.0]
    assert torch.equal(whole, stream)
    assert first_moved == 10


if __name__ == "__main__":
    main()
