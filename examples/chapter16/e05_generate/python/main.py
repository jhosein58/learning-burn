"""Example 16.5 - Generating text (PyTorch reference + parity test).

The same sampler as rust/src/main.rs -- temperature, top-k, and the inverse CDF at a
uniform number from the shared LCG -- so on the same weights both frameworks draw the
same characters. main() samples from the model that Example 16.4's main.py trained.

Run:   python main.py      (after python ../../e04_train/python/main.py)
Test:  pytest main.py
"""

import json
import math
import os
import tempfile
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GptConfig:
    vocab_size: int
    block_size: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0


class Gpt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.block_size, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)                # pre-norm
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layers,
                                            enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)

    def next_logits(self, tokens):
        n, t = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(t))
        causal = torch.ones(t, t, dtype=torch.bool).triu(1)
        x = self.blocks(x, mask=causal, is_causal=True)
        return self.head(self.ln_f(x[:, -1]))


class Lcg:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def unit(self):
        return self.next() / 32768.0

    def uniform(self, n, bound):
        return torch.tensor([(self.next() / 32768.0 * 2.0 - 1.0) * bound for _ in range(n)],
                            dtype=torch.float32)

    def linear(self, d_in, d_out, bias=True):
        bound = 1.0 / math.sqrt(d_in)
        w = self.uniform(d_out * d_in, bound).reshape(d_out, d_in)
        return w, (self.uniform(d_out, bound) if bias else None)

    def norm(self, ln):
        ln.weight.copy_(1.0 + self.uniform(ln.weight.numel(), 0.1))
        ln.bias.copy_(self.uniform(ln.bias.numel(), 0.1))


@torch.no_grad()
def with_fixed_weights(model, seed):
    v, d = model.tok.weight.shape
    t = model.pos.weight.shape[0]
    f = Lcg(seed)
    model.tok.weight.copy_(f.uniform(v * d, 0.1).reshape(v, d))
    model.pos.weight.copy_(f.uniform(t * d, 0.1).reshape(t, d))
    for layer in model.blocks.layers:
        f.norm(layer.norm1)
        (wq, bq), (wk, bk), (wv, bv) = (f.linear(d, d) for _ in range(3))
        layer.self_attn.in_proj_weight.copy_(torch.cat([wq, wk, wv]))
        layer.self_attn.in_proj_bias.copy_(torch.cat([bq, bk, bv]))
        w, b = f.linear(d, d)
        layer.self_attn.out_proj.weight.copy_(w)
        layer.self_attn.out_proj.bias.copy_(b)
        f.norm(layer.norm2)
        w, b = f.linear(d, 4 * d)
        layer.linear1.weight.copy_(w)
        layer.linear1.bias.copy_(b)
        w, b = f.linear(4 * d, d)
        layer.linear2.weight.copy_(w)
        layer.linear2.bias.copy_(b)
    f.norm(model.ln_f)
    w, _ = f.linear(d, v, bias=False)
    model.head.weight.copy_(w)
    return model


def sample(logits, temperature, top_k, u):
    """Temperature, top-k, softmax, inverse CDF at u. Temperature 0 is greedy."""
    if temperature == 0.0:
        return int(logits.argmax())
    top, ids = torch.topk(logits / temperature, top_k)          # largest first
    cdf = torch.softmax(top, 0).cumsum(0)
    pick = min(int((cdf < u).sum()), top_k - 1)
    return int(ids[pick])


@torch.no_grad()
def generate(model, prompt, new_tokens, temperature, top_k, seed):
    block = model.pos.weight.shape[0]
    rng = Lcg(seed)
    ids = list(prompt)
    for _ in range(new_tokens):
        context = torch.tensor([ids[-block:]])
        ids.append(sample(model.next_logits(context)[0], temperature, top_k, rng.unit()))
    return ids


def main():
    d = os.path.join(tempfile.gettempdir(), "learning-burn-c16-py")
    with open(os.path.join(d, "config.json")) as f:
        cfg = GptConfig(**json.load(f)["model"])
    vocab = open(os.path.join(d, "vocab.txt")).read()
    model = Gpt(cfg)
    model.load_state_dict(torch.load(os.path.join(d, "model.pt")))
    model.eval()
    print(f"loaded {sum(p.numel() for p in model.parameters())} parameters, "
          f"a vocabulary of {len(vocab)} symbols")
    prompt = [vocab.index(c) for c in "MENENIUS:\n"]
    with torch.no_grad():
        logits = model.next_logits(torch.tensor([prompt]))[0]
    print('next character after "MENENIUS:\\n", top 5:')
    for tau in (0.5, 1.0, 1.5):
        top, ids = torch.topk(torch.softmax(logits / tau, 0), 5)
        print(f"  tau {tau:.1f}: " + ", ".join(f"{vocab[i]!r} {p:.3f}"
                                               for i, p in zip(ids.tolist(), top.tolist())))
    for tau, k in ((0.0, 1), (0.8, 10), (1.0, len(vocab)), (1.5, len(vocab))):
        ids = generate(model, prompt, 160, tau, k, 2026)
        print(f"--- temperature {tau:.1f}, top-k {k} ---")
        print("".join(vocab[i] for i in ids))


def test_matches_burn():
    model = with_fixed_weights(Gpt(GptConfig(62)), 42).eval()
    got = generate(model, [16, 44, 53, 54, 55], 24, 0.8, 10, 7)       # "First"
    assert got == EXPECTED


EXPECTED = [16, 44, 53, 54, 55, 18, 34, 1, 6, 1, 28, 60, 33, 21, 6, 6, 18, 17, 34, 28, 28, 61, 23,
            23, 28, 28, 34, 28, 15]                          # from rust/src/main.rs


if __name__ == "__main__":
    main()
