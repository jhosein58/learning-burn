"""Example 17.2 - The noise-prediction loss and its gradient (PyTorch reference + parity test).

The same MLP, the same LCG weights and the same LCG batch as the Rust example
(which is also step 1 of Example 17.4). The gradient is computed twice: by hand,
with Chapter 9's dense-layer rule, and by torch .backward().

Run:   python main.py
Test:  pytest main.py
"""

import math

import numpy as np
import torch

R, SIGMA = 2.0, 0.15          # ring of 8 Gaussians
T = 100                       # diffusion steps
BETA_1, BETA_T = 1e-3, 0.2    # the DDPM schedule rescaled to T = 100
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


def linear_schedule():
    beta, alpha, abar = [0.0] * (T + 1), [1.0] * (T + 1), [1.0] * (T + 1)
    for t in range(1, T + 1):
        beta[t] = BETA_1 + (BETA_T - BETA_1) * (t - 1) / (T - 1)
        alpha[t] = 1.0 - beta[t]
        abar[t] = abar[t - 1] * alpha[t]
    return np.array(beta), np.array(alpha), np.array(abar)


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




def ddpm_batch(rng, abar):
    d = rng.take(10 * BATCH).reshape(BATCH, 10)
    x0 = f32(data_point(d[:, 0:5]))
    t = 1 + (d[:, 5] % T).astype(np.int64)
    eps = f32(gauss2(d[:, 6:10]))
    ca, cb = f32(np.sqrt(abar[t]))[:, None], f32(np.sqrt(1.0 - abar[t]))[:, None]
    xt = x0 * ca + eps * cb                       # the closed-form noising
    return torch.cat([xt, embed(t / T)], 1), eps, (x0[0], int(t[0]), eps[0])


def build():
    _, _, abar = linear_schedule()
    p = [w.requires_grad_() for w in init(Lcg(17))]
    w1, b1, w2, b2, w3, b3 = p
    h0, eps, row0 = ddpm_batch(Lcg(42), abar)
    # forward, keeping what the backward pass needs
    z1 = h0 @ w1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ w2 + b2
    a2 = torch.relu(z2)
    out = a2 @ w3 + b3
    diff = out - eps
    loss = (diff * diff).sum() / BATCH            # (1/N) sum ||eps_hat - eps||^2
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
    return row0, h0[0, :2], out[0].detach(), dout[0], loss.item(), hand, auto


def fmt(v):
    return "[" + ", ".join(f"{x:.6f}" for x in v.flatten().tolist()) + "]"


def main():
    row0, xt0, eps_hat0, dout0, loss, hand, auto = build()
    _, _, abar = linear_schedule()
    (x, t, e) = row0
    print(f"batch N = {BATCH}; row 0:")
    print(f"  x_0 = ({x[0]:.6f}, {x[1]:.6f})   t = {t}   eps = ({e[0]:.6f}, {e[1]:.6f})")
    print(f"  sqrt(abar_t) = {math.sqrt(abar[t]):.6f}   sqrt(1 - abar_t) = {math.sqrt(1 - abar[t]):.6f}")
    print(f"  x_t           = {fmt(xt0)}")
    print(f"  eps_hat       = {fmt(eps_hat0)}")
    print(f"  dL/d eps_hat  = {fmt(dout0)}")
    print(f"loss = {loss:.6f}")
    print(f"dL/db3         = {fmt(hand[5])}")
    print(f"dL/dW3[0, :]   = {fmt(hand[4][0])}")
    print(f"dL/db1[0..4]   = {fmt(hand[1][0, :4])}")
    diff = max((a - b).abs().max().item() for a, b in zip(hand, auto))
    count = sum(g.numel() for g in hand)
    print(f"torch autograd: max|hand - autograd| over all {count} gradients = {diff:.1e}")


# The Rust run's values (its matches_pytorch test pins them to these numbers).
RUST_LOSS = 1.9449306
RUST_B3 = [-0.010712076, -0.14410010]


def test_matches_burn():
    _, _, _, _, loss, hand, auto = build()
    assert max((a - b).abs().max().item() for a, b in zip(hand, auto)) < 1e-6
    assert abs(loss - RUST_LOSS) < 2e-6
    assert all(abs(a - b) < 2e-6 for a, b in zip(hand[5].flatten().tolist(), RUST_B3))


if __name__ == "__main__":
    main()
