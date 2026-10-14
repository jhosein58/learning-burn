"""Example 17.3 - One DDPM step and one DDIM step, with fixed inputs (PyTorch reference + parity test).

The network keeps its fixed LCG weights (untrained). The inputs are Example 17.1's
three points at t = 50. Then the same steps with an "oracle" eps_hat = the true eps.

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




def forward(p, h0):
    w1, b1, w2, b2, w3, b3 = p
    a1 = torch.relu(h0 @ w1 + b1)
    a2 = torch.relu(a1 @ w2 + b2)
    return a2 @ w3 + b3


def eps_model(p, x, t):
    return forward(p, torch.cat([x, embed(np.full(len(x), t / T))], 1))


def ddpm_step(sched, x, eps_hat, t, z):
    """One DDPM ancestral step t -> t-1, with fresh noise z."""
    beta, alpha, abar = sched
    mean = (x - eps_hat * (beta[t] / math.sqrt(1.0 - abar[t]))) / math.sqrt(alpha[t])
    var = beta[t] * (1.0 - abar[t - 1]) / (1.0 - abar[t])   # 0 at t = 1
    return mean + z * math.sqrt(var)


def ddim_step(sched, x, eps_hat, t, prev):
    """One DDIM step t -> prev (eta = 0). Returns (x0_hat, x_prev)."""
    abar = sched[2]
    x0_hat = (x - eps_hat * math.sqrt(1.0 - abar[t])) / math.sqrt(abar[t])
    return x0_hat, x0_hat * math.sqrt(abar[prev]) + eps_hat * math.sqrt(1.0 - abar[prev])


def build():
    sched = linear_schedule()
    beta, alpha, abar = sched
    t, prev = 50, 40
    x0 = f32(data_point(Lcg(42).take(5 * 3).reshape(3, 5)))
    eps = f32(gauss2(Lcg(7).take(4 * 3).reshape(3, 4)))
    xt = x0 * math.sqrt(abar[t]) + eps * math.sqrt(1.0 - abar[t])
    p = init(Lcg(17))
    with torch.no_grad():
        eps_hat = eps_model(p, xt, t)
    z = f32(gauss2(Lcg(8).take(4 * 3).reshape(3, 4)))
    x49 = ddpm_step(sched, xt, eps_hat, t, z)
    x0_hat, x40 = ddim_step(sched, xt, eps_hat, t, prev)
    # oracle: eps_hat = eps
    o_x0, o_x40 = ddim_step(sched, xt, eps, t, prev)
    x40_closed = x0 * math.sqrt(abar[prev]) + eps * math.sqrt(1.0 - abar[prev])
    o_mean = ddpm_step(sched, xt, eps, t, torch.zeros_like(eps))
    c0 = math.sqrt(abar[t - 1]) * beta[t] / (1.0 - abar[t])
    ct = math.sqrt(alpha[t]) * (1.0 - abar[t - 1]) / (1.0 - abar[t])
    posterior = x0 * c0 + xt * ct
    oracle = [(a - b).abs().max().item() for a, b in
              ((o_x0, x0), (o_x40, x40_closed), (o_mean, posterior))]
    return xt, eps_hat, z, x49, x0_hat, x40, oracle


def fmt(v):
    return " ".join(f"({a:.4f}, {b:.4f})" for a, b in v.tolist())


def main():
    xt, eps_hat, z, x49, x0_hat, x40, oracle = build()
    beta, alpha, abar = linear_schedule()
    t, prev = 50, 40
    print(f"x_50     = {fmt(xt)}")
    print(f"eps_hat  = {fmt(eps_hat)}   (fixed LCG weights, untrained)")
    print("DDPM step 50 -> 49:")
    var = beta[t] * (1.0 - abar[t - 1]) / (1.0 - abar[t])
    print(f"  1/sqrt(alpha_t) = {1 / math.sqrt(alpha[t]):.6f}   "
          f"beta_t/sqrt(1-abar_t) = {beta[t] / math.sqrt(1 - abar[t]):.6f}   sigma_t = {math.sqrt(var):.6f}")
    print(f"  z        = {fmt(z)}")
    print(f"  x_49     = {fmt(x49)}")
    print("DDIM step 50 -> 40:")
    print(f"  sqrt(abar_50) = {math.sqrt(abar[t]):.6f}   sqrt(abar_40) = {math.sqrt(abar[prev]):.6f}   "
          f"sqrt(1-abar_40) = {math.sqrt(1 - abar[prev]):.6f}")
    print(f"  x0_hat   = {fmt(x0_hat)}")
    print(f"  x_40     = {fmt(x40)}")
    print("oracle (eps_hat = the true eps), max |difference|:")
    print(f"  DDIM x0_hat vs x_0                    {oracle[0]:.1e}")
    print(f"  DDIM x_40 vs closed-form x_40         {oracle[1]:.1e}")
    print(f"  DDPM mean vs posterior mean mu~_50    {oracle[2]:.1e}")


# The Rust run's x_49 and x_40 (its matches_pytorch test pins them to these numbers).
RUST_X49 = [-0.42829746, 1.82075202, -0.25721037, -0.28487337, -1.08297169, 1.62019134]
RUST_X40 = [-0.13824841, 1.91570556, -0.49128374, -0.70782411, -2.27737999, 2.26543832]


def test_matches_burn():
    _, _, _, x49, _, x40, oracle = build()
    assert all(abs(a - b) < 1e-5 for a, b in zip(x49.flatten().tolist(), RUST_X49))
    assert all(abs(a - b) < 1e-5 for a, b in zip(x40.flatten().tolist(), RUST_X40))
    assert all(d < 1e-5 for d in oracle)


if __name__ == "__main__":
    main()
