"""Example 15.7 - A KV Cache by Hand (PyTorch reference + parity test).

Example 15.4's GPT with a 128-token context generates greedily twice: once
re-running the whole prefix every step, once with a cache of keys and values.
The logits must agree, and mm() counts the multiply-adds of every matmul.

Run:   python main.py
Test:  pytest main.py
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

V, T, D, H, L = 62, 128, 64, 4, 2
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


MACS = [0]


def mm(a, b):
    """matmul that counts its multiply-adds: (batch * m * k) * n."""
    MACS[0] += a.numel() * b.shape[-1]
    return a @ b


def layer_norm(x, g, b):
    mean = x.mean(dim=1, keepdim=True)
    xc = x - mean
    var = (xc ** 2).mean(dim=1, keepdim=True)
    return xc / (var + 1e-5).sqrt() * g + b


def attention(x, p, n, t):
    w_qkv, b_qkv, w_o, b_o = p
    dk = D // H
    qkv = mm(x, w_qkv) + b_qkv
    heads = lambda m: m.reshape(n, t, H, dk).transpose(1, 2)
    q, k, v = heads(qkv[:, :D]), heads(qkv[:, D:2 * D]), heads(qkv[:, 2 * D:])
    mask = torch.full((t, t), -1e9).triu(1).reshape(1, 1, t, t)
    scores = mm(q, k.transpose(2, 3)) * (1.0 / math.sqrt(dk)) + mask
    ctx = mm(torch.softmax(scores, dim=3), v)
    return mm(ctx.transpose(1, 2).reshape(n * t, D), w_o) + b_o


def block(x, p, n, t):
    ln1_g, ln1_b, _, _, _, _, ln2_g, ln2_b, w_fc, b_fc, w_pr, b_pr = p
    x = x + attention(layer_norm(x, ln1_g, ln1_b), p[2:6], n, t)
    h = F.gelu(mm(layer_norm(x, ln2_g, ln2_b), w_fc) + b_fc)
    return x + mm(h, w_pr) + b_pr


def forward(p, ids, n, t):
    tok = F.embedding(torch.tensor(ids), p[0])         # row lookup, [n*t, D]
    x = (tok.reshape(n, t, D) + p[1][:t].unsqueeze(0)).reshape(n * t, D)
    for l in range(L):
        x = block(x, p[2 + 12 * l:2 + 12 * (l + 1)], n, t)
    x = layer_norm(x, p[2 + 12 * L], p[3 + 12 * L])
    return mm(x, p[0].T)                               # tied head


def step(p, kv, idx, pos):
    dk = D // H
    x = p[0][idx:idx + 1] + p[1][pos:pos + 1]
    for l in range(L):
        blk = p[2 + 12 * l:2 + 12 * (l + 1)]
        ln1_g, ln1_b, w_qkv, b_qkv, w_o, b_o = blk[:6]
        ln2_g, ln2_b, w_fc, b_fc, w_pr, b_pr = blk[6:]
        qkv = mm(layer_norm(x, ln1_g, ln1_b), w_qkv) + b_qkv
        q = qkv[:, :D]
        for key, row in (("k", qkv[:, D:2 * D]), ("v", qkv[:, 2 * D:])):
            old = kv[key][l]
            kv[key][l] = row if old is None else torch.cat([old, row], 0)
        k, v = kv["k"][l], kv["v"][l]
        t = k.shape[0]
        q = q.reshape(1, H, dk).transpose(0, 1)
        k = k.reshape(t, H, dk).transpose(0, 1)
        v = v.reshape(t, H, dk).transpose(0, 1)
        scores = mm(q, k.transpose(1, 2)) * (1.0 / math.sqrt(dk))
        ctx = mm(torch.softmax(scores, dim=2), v).transpose(0, 1).reshape(1, D)
        x = x + mm(ctx, w_o) + b_o
        h = F.gelu(mm(layer_norm(x, ln2_g, ln2_b), w_fc) + b_fc)
        x = x + mm(h, w_pr) + b_pr
    x = layer_norm(x, p[2 + 12 * L], p[3 + 12 * L])
    return mm(x, p[0].T)[0]


def generate_full(p, prompt, new):
    MACS[0] = 0
    start = time.perf_counter()
    ids, logits = list(prompt), []
    for _ in range(new):
        last = forward(p, ids, 1, len(ids))[-1]
        ids.append(int(last.argmax()))
        logits.append(last)
    return ids, logits, MACS[0], (time.perf_counter() - start) * 1e3


def generate_cached(p, prompt, new):
    MACS[0] = 0
    start = time.perf_counter()
    cache = {"k": [None] * L, "v": [None] * L}
    ids, logits = list(prompt), []
    for pos, i in enumerate(prompt):
        last = step(p, cache, i, pos)
    for _ in range(new):
        ids.append(int(last.argmax()))
        logits.append(last)
        if len(ids) == len(prompt) + new:
            break
        last = step(p, cache, ids[-1], len(ids) - 1)
    return ids, logits, MACS[0], (time.perf_counter() - start) * 1e3


@torch.no_grad()
def build():
    voc = vocab()
    p = init_params(42)
    prompt = encode(b"CORIOLANUS:\n", voc)
    new = T - len(prompt)
    return generate_full(p, prompt, new), generate_cached(p, prompt, new), voc, len(prompt)


def max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a[1], b[1]))


def main():
    full, cached, voc, n_prompt = build()
    new = len(full[0]) - n_prompt
    print(f"context {T}, prompt {n_prompt} tokens, {new} new tokens, greedy")
    print("          multiply-adds      time")
    for name, g in (("no cache", full), ("KV cache", cached)):
        print(f"{name}  {g[2]:>13}  {g[3]:>6.1f} ms")
    print(f"ratio     {full[2] / cached[2]:>12.1f}x  {full[3] / cached[3]:>8.1f}x")
    print(f"max|logits(no cache) - logits(cache)| over {new} steps = {max_diff(full, cached):.1e}")
    print("same tokens:", str(full[0] == cached[0]).lower())
    text = bytes(voc[i] for i in cached[0][n_prompt:]).decode()
    print(f"generated (untrained weights): \"{text[:20]}\"")


def test_matches_burn():
    full, cached, _, _ = build()
    assert max_diff(full, cached) < 1e-4 and full[0] == cached[0]
    assert (full[2], cached[2]) == (1001252608, 15069312)
    assert full[0][12:22] == [28] * 10
    burn_last = [0.3557437, 0.4652831, -0.2058321, 0.1842696, 0.5877394]
    assert all(abs(a - b) < 1e-4 for a, b in zip(full[1][-1][:5].tolist(), burn_last))


if __name__ == "__main__":
    main()
