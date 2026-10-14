"""Example 15.4 - A Tiny GPT Forward Pass (PyTorch reference + parity test).

The same GPT as the Rust example, from raw tensors: token and position
embeddings, two pre-norm blocks of causal multi-head attention and a GELU MLP,
a final LayerNorm and a head tied to the token embedding. Same LCG weights,
same two input windows, so the logits and the loss must match.

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


def build():
    voc = vocab()
    assert len(voc) == V
    data = encode(TEXT, voc)
    p = init_params(42)
    n_params = sum(t.numel() for t in p)
    n = 2
    rng = Lcg(7)
    starts = [rng.below(len(data) - T - 1) for _ in range(n)]
    x = [i for s in starts for i in data[s:s + T]]
    y = [i for s in starts for i in data[s + 1:s + T + 1]]
    print("input 0  =", repr(bytes(voc[i] for i in x[:T]).decode()))
    logits = forward(p, x, n, T)
    loss = F.cross_entropy(logits, torch.tensor(y)).item()
    x2 = list(x)
    x2[T - 1] = (x2[T - 1] + 1) % V
    d = (forward(p, x2, n, T) - logits).abs()
    return n_params, list(logits.shape), logits[0, :5].tolist(), loss, \
        d[:T - 1].max().item(), d[T - 1].max().item()


def main():
    n_params, dims, first, loss, past, last = build()
    print(f"parameters = {n_params} in {len(layout())} tensors")
    print(f"logits     = {dims}  (n*t rows, one per position)")
    print("logits[0][..5] = [" + ", ".join(f"{v:.6f}" for v in first) + "]")
    print(f"loss       = {loss:.6f}  (ln V = {math.log(V):.6f})")
    print(f"change token 31 -> max|dlogits| at positions 0..30 = {past:.1e}")
    print(f"                   max|dlogits| at position 31     = {last:.4f}")


def test_matches_burn():
    n_params, dims, first, loss, past, last = build()
    assert n_params == 106112 and dims == [64, 62]
    burn = [0.303122, 0.397986, -0.074565, 0.320760, 0.163742]
    assert all(abs(a - b) < 1e-5 for a, b in zip(first, burn))
    assert abs(loss - 4.323092) < 1e-5
    assert past == 0.0 and last > 0.01


if __name__ == "__main__":
    main()
