"""Example 17.5 - Flow matching by hand (PyTorch reference + parity test).

The straight path x_t = (1 - t) x_0 + t x_1 from noise x_0 to data x_1, its
velocity x_1 - x_0, the loss and its gradient (by hand and by torch .backward()),
and one Euler and one Heun step, with the same LCG weights and inputs as Rust.

Run:   python main.py
Test:  pytest main.py
"""

import math

import numpy as np
import torch

R, SIGMA = 2.0, 0.15          # ring of 8 Gaussians
K = 8                         # time-embedding frequencies -> 16 features
IN, H = 2 + 2 * K, 128
BATCH = 256


class Lcg:
    """The LCG of Chapter 11, bit-identical to the Rust one, vectorised by jumping ahead."""
    A, C, MASK = 1103515245, 12345, 0xFFFFFFFF
    _a, _c = np.zeros(0, np.uint64), np.zeros(0, np.uint64)

    def __init__(self, seed):
        self.state = seed

    @classmethod
    def _tables(cls, n):
        # state_k = a_k * state_0 + c_k (mod 2^32), for k = 1..n
        if len(cls._a) < n:
            a, c = np.zeros(n, np.uint64), np.zeros(n, np.uint64)
            ak, ck = 1, 0
            for k in range(n):
                ak, ck = (ak * cls.A) & cls.MASK, (ck * cls.A + cls.C) & cls.MASK
                a[k], c[k] = ak, ck
            cls._a, cls._c = a, c
        return cls._a[:n], cls._c[:n]

    def take(self, n):
        """The next n 15-bit outputs."""
        a, c = self._tables(n)
        states = (a * np.uint64(self.state) + c) & np.uint64(self.MASK)
        self.state = int(states[-1])
        return ((states >> np.uint64(16)) & np.uint64(0x7FFF)).astype(np.float64)


def uniform(hi, lo):
    return (hi * 32768.0 + lo + 0.5) / 1073741824.0


def gauss2(d):
    """Box-Muller on draws d[:, 0:4] -> [n, 2] standard normals (float64)."""
    r = np.sqrt(-2.0 * np.log(uniform(d[:, 0], d[:, 1])))
    th = 2.0 * np.pi * uniform(d[:, 2], d[:, 3])
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def data_point(d):
    """d[:, 0:5] -> [n, 2] points of the ring of 8 Gaussians."""
    ang = 2.0 * np.pi * (d[:, 0] % 8) / 8.0
    g = gauss2(d[:, 1:5])
    return np.stack([R * np.cos(ang) + SIGMA * g[:, 0], R * np.sin(ang) + SIGMA * g[:, 1]], 1)


def embed(s):
    """sin(2^k s), cos(2^k s), k = 0..K-1, for an array s -> [n, 16] float32."""
    w = np.stack([float(1 << k) * s for k in range(K)], 1)
    return torch.from_numpy(np.concatenate([np.sin(w), np.cos(w)], 1).astype(np.float32))


def f32(a):
    return torch.from_numpy(np.asarray(a).astype(np.float32))


def init(rng):
    """nn.Linear's U(-1/sqrt(fan_in), 1/sqrt(fan_in)), drawn from the LCG. Weights are [in, out]."""
    def draw(n, fan_in, shape):
        bound = 1.0 / math.sqrt(fan_in)
        d = rng.take(2 * n).reshape(n, 2)
        return f32((2.0 * uniform(d[:, 0], d[:, 1]) - 1.0) * bound).reshape(shape)
    return [draw(IN * H, IN, (IN, H)), draw(H, IN, (1, H)),
            draw(H * H, H, (H, H)), draw(H, H, (1, H)),
            draw(H * 2, H, (H, 2)), draw(2, H, (1, 2))]




def forward(p, h0):
    w1, b1, w2, b2, w3, b3 = p
    a1 = torch.relu(h0 @ w1 + b1)
    a2 = torch.relu(a1 @ w2 + b2)
    return a2 @ w3 + b3


def fm_batch(rng):
    d = rng.take(11 * BATCH).reshape(BATCH, 11)
    x1 = f32(data_point(d[:, 0:5]))               # data
    t = uniform(d[:, 5], d[:, 6])
    x0 = f32(gauss2(d[:, 7:11]))                  # noise
    xt = x0 * f32(1.0 - t)[:, None] + x1 * f32(t)[:, None]    # the straight path
    return torch.cat([xt, embed(t)], 1), x1 - x0, (x1[0], float(t[0]), x0[0])


def euler_step(v, x, t, h):
    return x + v(x, t) * h


def heun_step(v, x, t, h):
    k1 = v(x, t)
    k2 = v(x + k1 * h, t + h)
    return x + (k1 + k2) * (h / 2.0)


def build():
    p = [w.requires_grad_() for w in init(Lcg(17))]
    w1, b1, w2, b2, w3, b3 = p
    h0, v, row0 = fm_batch(Lcg(42))
    z1 = h0 @ w1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ w2 + b2
    a2 = torch.relu(z2)
    out = a2 @ w3 + b3
    diff = out - v
    loss = (diff * diff).sum() / BATCH            # (1/N) sum ||v_hat - (x_1 - x_0)||^2
    with torch.no_grad():                         # ---- (1) by hand ----
        dout = diff * (2.0 / BATCH)
        gw3, gb3 = a2.T @ dout, dout.sum(0, keepdim=True)
        d2 = (dout @ w3.T) * (z2 > 0)
        gw2, gb2 = a1.T @ d2, d2.sum(0, keepdim=True)
        d1 = (d2 @ w2.T) * (z1 > 0)
        gw1, gb1 = h0.T @ d1, d1.sum(0, keepdim=True)
    loss.backward()                               # ---- (2) autograd ----
    hand = [gw1, gb1, gw2, gb2, gw3, gb3]
    auto = [w.grad for w in p]
    first = [h0[0, :2], v[0], out[0].detach(), dout[0]]
    # one Euler and one Heun step, t = 0.5 -> 0.6
    x1 = f32(data_point(Lcg(42).take(5 * 3).reshape(3, 5)))
    x0 = f32(gauss2(Lcg(7).take(4 * 3).reshape(3, 4)))
    x = (x0 + x1) * 0.5
    q = [w.detach() for w in p]
    net = lambda x, t: forward(q, torch.cat([x, embed(np.full(len(x), t))], 1))
    with torch.no_grad():
        euler, heun = euler_step(net, x, 0.5, 0.1), heun_step(net, x, 0.5, 0.1)
    straight = lambda x, t: x1 - x0
    on_path = x0 * 0.4 + x1 * 0.6
    oracle = [(euler_step(straight, x, 0.5, 0.1) - on_path).abs().max().item(),
              (heun_step(straight, x, 0.5, 0.1) - on_path).abs().max().item()]
    return row0, first, loss.item(), hand, auto, [x, euler, heun], oracle


def fmt(v):
    return "[" + ", ".join(f"{x:.6f}" for x in v.flatten().tolist()) + "]"


def fmt_pts(v):
    return " ".join(f"({a:.4f}, {b:.4f})" for a, b in v.tolist())


def main():
    row0, first, loss, hand, auto, steps, oracle = build()
    x1, t, x0 = row0
    print(f"batch N = {BATCH}; row 0:")
    print(f"  x_1 (data) = ({x1[0]:.6f}, {x1[1]:.6f})   t = {t:.6f}   x_0 (noise) = ({x0[0]:.6f}, {x0[1]:.6f})")
    print(f"  x_t            = {fmt(first[0])}")
    print(f"  x_1 - x_0      = {fmt(first[1])}")
    print(f"  v_hat          = {fmt(first[2])}")
    print(f"  dL/d v_hat     = {fmt(first[3])}")
    print(f"loss = {loss:.6f}")
    print(f"dL/db3         = {fmt(hand[5])}")
    print(f"dL/dW3[0, :]   = {fmt(hand[4][0])}")
    print(f"dL/db1[0..4]   = {fmt(hand[1][0, :4])}")
    diff = max((a - b).abs().max().item() for a, b in zip(hand, auto))
    print(f"torch autograd: max|hand - autograd| over all {sum(g.numel() for g in hand)} gradients = {diff:.1e}")
    print("one step of size h = 0.1, from t = 0.5 (fixed LCG weights, untrained):")
    print(f"  x_0.5          = {fmt_pts(steps[0])}")
    print(f"  Euler  x_0.6   = {fmt_pts(steps[1])}   (1 network call)")
    print(f"  Heun   x_0.6   = {fmt_pts(steps[2])}   (2 network calls)")
    print(f"oracle v = x_1 - x_0, max |x_0.6 - (0.4 x_0 + 0.6 x_1)|: Euler {oracle[0]:.1e}  Heun {oracle[1]:.1e}")


# The Rust run's values (its matches_pytorch test pins them to these numbers).
RUST_LOSS = 6.0920348
RUST_EULER = [0.47371492, 1.06464696, -0.72520512, -0.77912533, -1.46242058, 0.78795898]
RUST_HEUN = [0.47029307, 1.06239915, -0.72773290, -0.78301638, -1.46555471, 0.78503275]


def test_matches_burn():
    _, _, loss, hand, auto, steps, oracle = build()
    assert max((a - b).abs().max().item() for a, b in zip(hand, auto)) < 1e-6
    assert abs(loss - RUST_LOSS) < 1e-5
    assert all(abs(a - b) < 1e-5 for a, b in zip(steps[1].flatten().tolist(), RUST_EULER))
    assert all(abs(a - b) < 1e-5 for a, b in zip(steps[2].flatten().tolist(), RUST_HEUN))
    assert all(d < 1e-6 for d in oracle)


if __name__ == "__main__":
    main()
