"""Example 15.5 - Training the Tiny GPT with a Hand-written Adam (PyTorch reference).

Example 15.4's GPT, trained on the Shakespeare slice with Adam written out
(Chapter 7's equations, bias correction included). Same LCG weights, same LCG
batches, so the first losses must match the Rust run; later losses drift
apart only by float rounding. Then a short sample at temperature 0.8.

Run:   python main.py
Test:  pytest main.py
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

V, T, D, H, L = 62, 32, 64, 4, 2
FF = 4 * D
torch.set_num_threads(1)  # tensors this small run fastest on one thread
TEXT = (Path(__file__).resolve().parents[2] / "data" / "shakespeare.txt").read_bytes()


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


def vocab():
    return sorted(set(TEXT))


def encode(text, voc):
    index = {b: i for i, b in enumerate(voc)}
    return [index[b] for b in text]


def layout():
    s = np.float32(1.0 / math.sqrt(D))
    sf = np.float32(1.0 / math.sqrt(FF))
    p = [("wte", (V, D), ("u", np.float32(0.1))), ("wpe", (T, D), ("u", np.float32(0.1)))]
    for l in range(L):
        blk = [
            ("ln1.g", (1, D), ("1",)), ("ln1.b", (1, D), ("0",)),
            ("attn.w_qkv", (D, 3 * D), ("u", s)), ("attn.b_qkv", (1, 3 * D), ("0",)),
            ("attn.w_o", (D, D), ("u", s)), ("attn.b_o", (1, D), ("0",)),
            ("ln2.g", (1, D), ("1",)), ("ln2.b", (1, D), ("0",)),
            ("mlp.w_fc", (D, FF), ("u", s)), ("mlp.b_fc", (1, FF), ("0",)),
            ("mlp.w_pr", (FF, D), ("u", sf)), ("mlp.b_pr", (1, D), ("0",)),
        ]
        p += [(f"h{l}.{n}", sh, i) for n, sh, i in blk]
    p += [("lnf.g", (1, D), ("1",)), ("lnf.b", (1, D), ("0",))]
    return p


def init_params(seed):
    rng = Lcg(seed)
    params = []
    for _, (r, c), init in layout():
        if init[0] == "u":
            u = np.array([rng.uniform() for _ in range(r * c)], dtype=np.float32)
            vals = (np.float32(2.0) * u - np.float32(1.0)) * init[1]
        elif init[0] == "1":
            vals = np.ones(r * c, dtype=np.float32)
        else:
            vals = np.zeros(r * c, dtype=np.float32)
        params.append(torch.tensor(vals.reshape(r, c)))
    return params


def layer_norm(x, g, b):
    mean = x.mean(dim=1, keepdim=True)
    xc = x - mean
    var = (xc ** 2).mean(dim=1, keepdim=True)
    return xc / (var + 1e-5).sqrt() * g + b


def attention(x, p, n, t):
    w_qkv, b_qkv, w_o, b_o = p
    dk = D // H
    qkv = x @ w_qkv + b_qkv
    heads = lambda m: m.reshape(n, t, H, dk).transpose(1, 2)
    q, k, v = heads(qkv[:, :D]), heads(qkv[:, D:2 * D]), heads(qkv[:, 2 * D:])
    mask = torch.full((t, t), -1e9).triu(1).reshape(1, 1, t, t)
    scores = q @ k.transpose(2, 3) * (1.0 / math.sqrt(dk)) + mask
    ctx = torch.softmax(scores, dim=3) @ v
    return ctx.transpose(1, 2).reshape(n * t, D) @ w_o + b_o


def block(x, p, n, t):
    ln1_g, ln1_b, _, _, _, _, ln2_g, ln2_b, w_fc, b_fc, w_pr, b_pr = p
    x = x + attention(layer_norm(x, ln1_g, ln1_b), p[2:6], n, t)
    h = F.gelu(layer_norm(x, ln2_g, ln2_b) @ w_fc + b_fc)
    return x + h @ w_pr + b_pr


def forward(p, ids, n, t):
    tok = F.embedding(torch.tensor(ids), p[0])         # row lookup, [n*t, D]
    x = (tok.reshape(n, t, D) + p[1][:t].unsqueeze(0)).reshape(n * t, D)
    for l in range(L):
        x = block(x, p[2 + 12 * l:2 + 12 * (l + 1)], n, t)
    x = layer_norm(x, p[2 + 12 * L], p[3 + 12 * L])
    return x @ p[0].T                                  # tied head


N, STEPS, EVERY = 16, 2000, 400
LR, BETA1, BETA2, EPS = 3e-3, 0.9, 0.999, 1e-8


def batch(data, rng):
    starts = [rng.below(len(data) - T - 1) for _ in range(N)]
    x = [i for s in starts for i in data[s:s + T]]
    y = [i for s in starts for i in data[s + 1:s + T + 1]]
    return x, y


class Adam:
    """Chapter 7's Adam, written out, one tensor at a time."""

    def __init__(self, params):
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    @torch.no_grad()
    def step(self, params):
        self.t += 1
        bc1, bc2 = 1 - BETA1 ** self.t, 1 - BETA2 ** self.t
        for i, p in enumerate(params):
            g = p.grad
            self.m[i] = BETA1 * self.m[i] + (1 - BETA1) * g
            self.v[i] = BETA2 * self.v[i] + (1 - BETA2) * g ** 2
            update = (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + EPS)
            p -= LR * update
            p.grad = None


@torch.no_grad()
def evaluate(p, val):
    return sum(F.cross_entropy(forward(p, x, N, T), torch.tensor(y)).item()
               for x, y in val) / len(val)


@torch.no_grad()
def generate(p, prompt, new, temp, seed):
    rng = Lcg(seed)
    ids = list(prompt)
    for _ in range(new):
        ctx = ids[-T:]
        logits = forward(p, ctx, 1, len(ctx))[-1]
        probs = torch.softmax(logits / temp, dim=0).tolist()
        u, cdf, nxt = rng.uniform(), 0.0, V - 1
        for i, q in enumerate(probs):
            cdf += q
            if u < cdf:
                nxt = i
                break
        ids.append(nxt)
    return ids


def build(steps):
    voc = vocab()
    data = encode(TEXT, voc)
    n_train = len(data) * 9 // 10
    train, valid = data[:n_train], data[n_train:]
    vrng = Lcg(1)
    val = [batch(valid, vrng) for _ in range(4)]
    p = [t.requires_grad_() for t in init_params(42)]
    adam = Adam(p)
    rng = Lcg(2024)
    log = [(0, float("nan"), evaluate(p, val))]
    first, window = [], 0.0
    for step in range(1, steps + 1):
        x, y = batch(train, rng)
        loss = F.cross_entropy(forward(p, x, N, T), torch.tensor(y))
        loss.backward()
        adam.step(p)
        if step <= 3:
            first.append(loss.item())
        window += loss.item()
        if step % EVERY == 0:
            log.append((step, window / EVERY, evaluate(p, val)))
            window = 0.0
    ids = generate(p, encode(b"CORIOLANUS:\n", voc), 200, 0.8, 99)
    return first, log, bytes(voc[i] for i in ids).decode()


def main():
    first, log, sample = build(STEPS)
    print(f"vocab {V}, context {T}, width {D}, {H} heads, {L} blocks, batch {N}x{T}")
    print("steps 1-3 loss = [" + ", ".join(f"{v:.6f}" for v in first) + "]")
    for step, train, val in log:
        if step == 0:
            print(f"step {step:>4}  val {val:.4f}")
        else:
            print(f"step {step:>4}  train {train:.4f}  val {val:.4f}")
    print("--- sample (temperature 0.8) ---")
    print(sample)


def test_matches_burn():
    first, _, _ = build(3)
    burn = [4.289678, 3.761544, 3.687974]
    assert all(abs(a - b) < 1e-4 for a, b in zip(first, burn))


if __name__ == "__main__":
    main()
