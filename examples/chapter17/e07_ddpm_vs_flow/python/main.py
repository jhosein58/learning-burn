"""Example 17.7 - DDPM and flow matching, head to head (PyTorch reference + parity test).

The same data, network, LCG weights, Adam and number of steps for both. Training
time, sample quality along the way at 20 network calls, the discretisation error
of coarse solves, and how straight the paths are.

Run:   python main.py
Test:  pytest main.py
"""

import math
import time

import numpy as np
import torch

torch.set_num_threads(1)  # matrices this small run fastest on one thread

R, SIGMA = 2.0, 0.15          # ring of 8 Gaussians
T = 100                       # diffusion steps
BETA_1, BETA_T = 1e-3, 0.2    # the DDPM schedule rescaled to T = 100
K = 8                         # time-embedding frequencies -> 16 features
IN, H = 2 + 2 * K, 128
BATCH, STEPS, LR = 256, 6000, 2e-3
SAMPLES, PATHS = 2000, 64


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


def ddpm_batch(rng, abar):
    d = rng.take(10 * BATCH).reshape(BATCH, 10)
    x0 = f32(data_point(d[:, 0:5]))
    t = 1 + (d[:, 5] % T).astype(np.int64)
    eps = f32(gauss2(d[:, 6:10]))
    ca, cb = f32(np.sqrt(abar[t]))[:, None], f32(np.sqrt(1.0 - abar[t]))[:, None]
    xt = x0 * ca + eps * cb                       # the closed-form noising
    return torch.cat([xt, embed(t / T)], 1), eps


def fm_batch(rng):
    d = rng.take(11 * BATCH).reshape(BATCH, 11)
    x1 = f32(data_point(d[:, 0:5]))               # data
    t = uniform(d[:, 5], d[:, 6])
    x0 = f32(gauss2(d[:, 7:11]))                  # noise
    xt = x0 * f32(1.0 - t)[:, None] + x1 * f32(t)[:, None]    # the straight path
    return torch.cat([xt, embed(t)], 1), x1 - x0


def noise(rng, n):
    return f32(gauss2(rng.take(4 * n).reshape(n, 4)))


def eps_model(p, x, t):
    return forward(p, torch.cat([x, embed(np.full(len(x), t / T))], 1))


def ddim_step(sched, x, eps_hat, t, prev):
    """One DDIM step t -> prev (eta = 0). Returns (x0_hat, x_prev)."""
    abar = sched[2]
    x0_hat = (x - eps_hat * math.sqrt(1.0 - abar[t])) / math.sqrt(abar[t])
    return x0_hat, x0_hat * math.sqrt(abar[prev]) + eps_hat * math.sqrt(1.0 - abar[prev])


def euler_step(v, x, t, h):
    return x + v(x, t) * h


def heun_step(v, x, t, h):
    k1 = v(x, t)
    k2 = v(x + k1 * h, t + h)
    return x + (k1 + k2) * (h / 2.0)


def data_set(seed, n):
    return data_point(Lcg(seed).take(5 * n).reshape(n, 5))


def hist_tv(a, b):
    def hist(pts):
        ij = np.floor((pts + 3.0) / 0.25)
        inside = ((ij >= 0) & (ij < 24)).all(1)
        idx = np.where(inside, ij[:, 0] * 24 + ij[:, 1], 24 * 24).astype(np.int64)
        return np.bincount(idx, minlength=24 * 24 + 1) / len(pts)
    return 0.5 * np.abs(hist(a) - hist(b)).sum()


def train(kind, steps, checkpoints=()):
    """kind is "ddpm" (target: the noise) or "flow" (target: the velocity x_1 - x_0)."""
    _, _, abar = linear_schedule()
    p = [w.requires_grad_() for w in init(Lcg(17))]
    opt = torch.optim.Adam(p, lr=LR, betas=(0.9, 0.999), eps=1e-8)
    rng = Lcg(42)
    saved, losses = [], []
    start = time.perf_counter()
    for k in range(steps):
        h0, target = ddpm_batch(rng, abar)[:2] if kind == "ddpm" else fm_batch(rng)
        diff = forward(p, h0) - target
        loss = (diff * diff).sum() / BATCH
        opt.zero_grad()
        loss.backward()
        for g in opt.param_groups:
            g["lr"] = LR * (1.0 - k / STEPS)
        opt.step()
        losses.append(loss.item())
        if k + 1 in checkpoints:
            saved.append([w.detach().clone() for w in p])
    return saved, losses, time.perf_counter() - start


@torch.no_grad()
def ddim_path(p, sched, steps, n):
    xs = [noise(Lcg(7), n)]
    for i in range(steps, 0, -1):
        t, prev = i * T // steps, (i - 1) * T // steps
        xs.append(ddim_step(sched, xs[-1], eps_model(p, xs[-1], t), t, prev)[1])
    return xs


@torch.no_grad()
def flow_path(p, steps, heun, n):
    v = lambda x, t: forward(p, torch.cat([x, embed(np.full(len(x), t))], 1))
    h = 1.0 / steps
    xs = [noise(Lcg(7), n)]
    for i in range(steps):
        xs.append((heun_step if heun else euler_step)(v, xs[-1], i * h, h))
    return xs


def straightness(xs):
    pts = [x.double() for x in xs]
    length = sum((b - a).norm(dim=1).sum().item() for a, b in zip(pts[:-1], pts[1:]))
    return length / (pts[-1] - pts[0]).norm(dim=1).sum().item()


def length_share(xs):
    seg = [(b.double() - a.double()).norm(dim=1).sum().item() for a, b in zip(xs[:-1], xs[1:])]
    return sum(seg[:10]) / sum(seg), sum(seg[-10:]) / sum(seg)


def rms(a, b):
    return math.sqrt(((a.double() - b.double()) ** 2).sum(1).mean().item())


def training_pairs(n):
    d = Lcg(42).take(11 * n).reshape(n, 11)
    return [f32(gauss2(d[:, 7:11])), f32(data_point(d[:, 0:5]))]


def main():
    sched = linear_schedule()
    checkpoints = (250, 500, 1000, 2000, 4000, 6000)
    ddpm, ddpm_losses, ddpm_secs = train("ddpm", STEPS, checkpoints)
    flow, flow_losses, flow_secs = train("flow", STEPS, checkpoints)
    ms = lambda secs: 1000 * secs / STEPS
    print(f"training, {STEPS} steps each, batch {BATCH}, the same network and Adam:")
    print(f"  DDPM            {ms(ddpm_secs):.1f} ms/step   mean loss of the last 1000 steps {np.mean(ddpm_losses[-1000:]):.4f}")
    print(f"  flow matching   {ms(flow_secs):.1f} ms/step   mean loss of the last 1000 steps {np.mean(flow_losses[-1000:]):.4f}")
    reference = data_set(99, SAMPLES)
    tv = lambda xs: hist_tv(xs[-1].double().numpy(), reference)
    print("tv along the way, 20 network calls per sample (data vs data: 0.093):")
    print("   step   DDIM 20 steps   Euler 20 steps   Heun 10 steps")
    for k, step in enumerate(checkpoints):
        a = tv(ddim_path(ddpm[k], sched, 20, SAMPLES))
        b = tv(flow_path(flow[k], 20, False, SAMPLES))
        c = tv(flow_path(flow[k], 10, True, SAMPLES))
        print(f"  {step:>5}   {a:.3f}           {b:.3f}            {c:.3f}")
    p, q = ddpm[-1], flow[-1]
    fine_ddim = ddim_path(p, sched, 100, SAMPLES)[-1]
    fine_flow = flow_path(q, 100, True, SAMPLES)[-1]
    print("RMS distance of the end point from a 100-step solve (DDIM 100; Heun 100 for the flow):")
    print("  steps    DDIM     Euler    Heun")
    for steps in (5, 10, 20, 50):
        a = rms(ddim_path(p, sched, steps, SAMPLES)[-1], fine_ddim)
        b = rms(flow_path(q, steps, False, SAMPLES)[-1], fine_flow)
        c = rms(flow_path(q, steps, True, SAMPLES)[-1], fine_flow)
        print(f"  {steps:>5}    {a:.4f}   {b:.4f}   {c:.4f}")
    ddim, euler = ddim_path(p, sched, 100, PATHS), flow_path(q, 100, False, PATHS)
    print(f"total path length / total start-to-end distance, {PATHS} paths, 100 steps:")
    print(f"  DDIM (deterministic DDPM)             {straightness(ddim):.3f}")
    print(f"  flow matching (Euler)                 {straightness(euler):.3f}")
    print(f"  flow-matching training pairs x_0 -> x_1  {straightness(training_pairs(PATHS)):.3f}")
    print("share of the path length covered in the first and the last 10 of the 100 steps:")
    for name, xs in (("DDIM         ", ddim), ("flow matching", euler)):
        first, last = length_share(xs)
        print(f"  {name}   first {first:.3f}   last {last:.3f}")


# The Rust run's first three losses of each kind (its matches_pytorch test pins them).
RUST_DDPM = [1.944931, 1.766543, 1.826159]
RUST_FLOW = [6.092034, 5.722399, 5.895596]


def test_matches_burn():
    _, ddpm, _ = train("ddpm", 3)
    _, flow, _ = train("flow", 3)
    assert all(abs(a - b) < 1e-5 for a, b in zip(ddpm + flow, RUST_DDPM + RUST_FLOW))


if __name__ == "__main__":
    main()
