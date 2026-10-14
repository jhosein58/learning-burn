"""Example 17.1 - The data, the noise schedule and the forward process (PyTorch reference + parity test).

2000 points from a ring of 8 Gaussians, drawn by the same LCG as the Rust example.
The closed-form noising x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, with fixed
noise, and the step-by-step chain checked against it.

Run:   python main.py
Test:  pytest main.py
"""

import math

import numpy as np
import torch

R, SIGMA = 2.0, 0.15          # ring of 8 Gaussians
T = 100                       # diffusion steps
BETA_1, BETA_T = 1e-3, 0.2    # the DDPM schedule rescaled to T = 100
N = 2000


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
    return beta, alpha, abar


def cosine_abar():
    f = lambda t: math.cos((t / T + 0.008) / 1.008 * math.pi / 2.0) ** 2
    abar = [1.0] * (T + 1)
    for t in range(1, T + 1):
        abar[t] = abar[t - 1] * (1.0 - min(1.0 - f(t) / f(t - 1), 0.999))
    return abar


def f32(a):
    return torch.from_numpy(np.asarray(a).astype(np.float32))


def noised(x0, eps, abar):
    """The closed form: one jump from x_0 to x_t."""
    return x0 * math.sqrt(abar) + eps * math.sqrt(1.0 - abar)


def mean_std(x):
    v = x.double()
    return v.mean(0).tolist(), v.std(0, unbiased=False).tolist()


def build():
    beta, alpha, abar = linear_schedule()
    x = f32(data_point(Lcg(42).take(5 * N).reshape(N, 5)))
    data_mean, data_std = mean_std(x)
    x0 = x[:3]
    eps = f32(gauss2(Lcg(7).take(4 * 3).reshape(3, 4)))
    jumps = [(t, noised(x0, eps, abar[t])) for t in (10, 50, 100)]
    rng = Lcg(8)
    xt, chain = x.clone(), []
    for t in range(1, T + 1):
        e = f32(gauss2(rng.take(4 * N).reshape(N, 4)))
        xt = xt * math.sqrt(alpha[t]) + e * math.sqrt(beta[t])   # one small step
        if t in (10, 25, 50, 75, 100):
            chain.append((t, *mean_std(xt)))
    return x0, eps, jumps, chain, data_mean, data_std


def fmt(v):
    return " ".join(f"({a:.4f}, {b:.4f})" for a, b in v.tolist())


def main():
    x0, eps, jumps, chain, data_mean, data_std = build()
    beta, _, abar = linear_schedule()
    cos = cosine_abar()
    print(f"data: {N} points, ring of 8 Gaussians, radius {R}, sigma {SIGMA}")
    print(f"  mean ({data_mean[0]:.4f}, {data_mean[1]:.4f})  std ({data_std[0]:.4f}, {data_std[1]:.4f})")
    print(f"schedule: linear beta from {BETA_1} to {BETA_T}, T = {T}")
    print("    t   beta_t  sqrt(abar_t)  sqrt(1-abar_t)  abar_t     abar_t (cosine)")
    for t in (1, 10, 25, 50, 75, 90, 100):
        print(f"  {t:>3}   {beta[t]:.4f}  {math.sqrt(abar[t]):.6f}      "
              f"{math.sqrt(1 - abar[t]):.6f}        {abar[t]:.3e}  {cos[t]:.3e}")
    print("closed form, three points, fixed noise:")
    print(f"  x_0    = {fmt(x0)}")
    print(f"  eps    = {fmt(eps)}")
    for t, v in jumps:
        print(f"  x_{t:<3}  = {fmt(v)}")
    print(f"the chain, step by step ({N} points), against the closed form:")
    print("    t   mean (chain)        std (chain)       std (closed form)")
    for t, m, sd in chain:
        pred = [math.sqrt(abar[t] * d * d + 1.0 - abar[t]) for d in data_std]
        print(f"  {t:>3}   ({m[0]:>7.4f}, {m[1]:>7.4f})  ({sd[0]:.4f}, {sd[1]:.4f})  "
              f"({pred[0]:.4f}, {pred[1]:.4f})")


# x_100 of the three points, and the chain's std after 100 steps: the Rust run's values
# (its matches_pytorch test pins them to these numbers).
RUST_X100 = [-0.49216002, 0.89037782, 0.11220753, -0.02384296, -0.92908782, 1.40687919]
RUST_STD100 = [0.99976702, 0.99686945]


def test_matches_burn():
    _, _, jumps, chain, _, _ = build()
    assert all(abs(a - b) < 2e-6 for a, b in zip(jumps[2][1].flatten().tolist(), RUST_X100))
    assert all(abs(a - b) < 1e-5 for a, b in zip(chain[4][2], RUST_STD100))


if __name__ == "__main__":
    main()
