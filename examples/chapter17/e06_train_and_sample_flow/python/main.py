"""Example 17.6 - Train flow matching, then integrate the ODE (PyTorch reference + parity test).

The same MLP, LCG weights, Adam and budget as Example 17.4; only the target changes,
from the noise to the velocity x_1 - x_0 of the straight path. Then dx/dt = v(x, t)
from t = 0 (noise) to t = 1 (data) with Euler and with Heun, from the same LCG noise.

Run:   python main.py
Test:  pytest main.py
"""

import math
import time

import numpy as np
import torch

torch.set_num_threads(1)  # matrices this small run fastest on one thread

R, SIGMA = 2.0, 0.15          # ring of 8 Gaussians
K = 8                         # time-embedding frequencies -> 16 features
IN, H = 2 + 2 * K, 128
BATCH, STEPS, LR = 256, 6000, 2e-3
SAMPLES = 2000


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
    return torch.cat([xt, embed(t)], 1), x1 - x0


def train(steps):
    p = [w.requires_grad_() for w in init(Lcg(17))]
    opt = torch.optim.Adam(p, lr=LR, betas=(0.9, 0.999), eps=1e-8)
    rng = Lcg(42)
    losses = []
    for k in range(steps):
        h0, v = fm_batch(rng)
        diff = forward(p, h0) - v
        loss = (diff * diff).sum() / BATCH       # (1/N) sum ||v_hat - (x_1 - x_0)||^2
        opt.zero_grad()
        loss.backward()
        for g in opt.param_groups:
            g["lr"] = LR * (1.0 - k / STEPS)
        opt.step()
        losses.append(loss.item())
    return [w.detach() for w in p], losses


def noise(rng, n):
    return f32(gauss2(rng.take(4 * n).reshape(n, 4)))


def euler_step(v, x, t, h):
    return x + v(x, t) * h


def heun_step(v, x, t, h):
    k1 = v(x, t)
    k2 = v(x + k1 * h, t + h)
    return x + (k1 + k2) * (h / 2.0)


@torch.no_grad()
def flow_sample(p, steps, heun):
    v = lambda x, t: forward(p, torch.cat([x, embed(np.full(len(x), t))], 1))
    h = 1.0 / steps
    x = noise(Lcg(7), SAMPLES)                     # the same starting noise as Example 17.4
    for i in range(steps):
        x = (heun_step if heun else euler_step)(v, x, i * h, h)
    return x


def data_set(seed, n):
    return data_point(Lcg(seed).take(5 * n).reshape(n, 5))


def hist_tv(a, b):
    def hist(pts):
        ij = np.floor((pts + 3.0) / 0.25)
        inside = ((ij >= 0) & (ij < 24)).all(1)
        idx = np.where(inside, ij[:, 0] * 24 + ij[:, 1], 24 * 24).astype(np.int64)
        return np.bincount(idx, minlength=24 * 24 + 1) / len(pts)
    return 0.5 * np.abs(hist(a) - hist(b)).sum()


def nn_dist(a, b):
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return d.min(1).mean()


def main():
    start = time.perf_counter()
    p, losses = train(STEPS)
    secs = time.perf_counter() - start
    for k, l in enumerate(losses[:5]):
        print(f"step {k + 1}  loss {l:.6f}")
    for c in range(0, STEPS, 1000):
        print(f"steps {c + 1:>4}-{c + 1000:<4}  mean loss {np.mean(losses[c:c + 1000]):.4f}")
    print(f"trained {STEPS} steps in {secs:.1f} s ({1000 * secs / STEPS:.1f} ms/step)")
    reference, other = data_set(99, SAMPLES), data_set(100, SAMPLES)
    print(f"floor (data vs data)   tv {hist_tv(other, reference):.3f}  nn {nn_dist(other, reference):.3f}")
    print("sampler        NFE   tv     nn")
    for name, heun in (("Euler", False), ("Heun ", True)):
        for steps in (1, 2, 5, 10, 20, 50, 100):
            x = flow_sample(p, steps, heun).double().numpy()
            nfe = 2 * steps if heun else steps
            print(f"{name} {steps:>3} steps {nfe:>4}  {hist_tv(x, reference):.3f}  {nn_dist(x, reference):.3f}")


# The first five losses of the Rust run, whose gradients and Adam are written by hand.
RUST_LOSSES = [6.092034, 5.722399, 5.895596, 5.441597, 6.121362]


def test_matches_burn():
    _, losses = train(5)
    assert all(abs(a - b) < 1e-5 for a, b in zip(losses, RUST_LOSSES)), losses


if __name__ == "__main__":
    main()
