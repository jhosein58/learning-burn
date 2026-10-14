"""Example 16.2 - A GPT from torch.nn (PyTorch reference + parity test).

The same architecture as rust/src/main.rs -- token and position embeddings, pre-norm
nn.TransformerEncoder layers with a causal mask, a final LayerNorm and a linear head --
with the same LCG weights, so the logits must agree.

Run:   python main.py
Test:  pytest main.py
"""

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                    "shakespeare.txt")


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

    def forward(self, tokens):
        n, t = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(t))
        causal = torch.ones(t, t, dtype=torch.bool).triu(1)        # True = hidden
        x = self.blocks(x, mask=causal, is_causal=True)
        return self.head(self.ln_f(x))


class Lcg:
    """Tiny LCG, bit-identical to the Rust one."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def uniform(self, n, bound):
        return torch.tensor([(self.next() / 32768.0 * 2.0 - 1.0) * bound for _ in range(n)],
                            dtype=torch.float32)

    def linear(self, d_in, d_out, bias=True):          # PyTorch layout: [out, in]
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
        f.norm(layer.norm1)                            # PyTorch's norm1 is Burn's norm_2
        (wq, bq), (wk, bk), (wv, bv) = (f.linear(d, d) for _ in range(3))
        layer.self_attn.in_proj_weight.copy_(torch.cat([wq, wk, wv]))
        layer.self_attn.in_proj_bias.copy_(torch.cat([bq, bk, bv]))
        w, b = f.linear(d, d)
        layer.self_attn.out_proj.weight.copy_(w)
        layer.self_attn.out_proj.bias.copy_(b)
        f.norm(layer.norm2)                            # ...and norm2 is Burn's norm_1
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


def windows():
    text = open(DATA).read()
    chars = sorted(set(text))
    lookup = {c: i for i, c in enumerate(chars)}
    tokens = torch.tensor([lookup[c] for c in text])
    t = 128
    inputs = torch.stack([tokens[0:t], tokens[t:2 * t]])
    targets = torch.stack([tokens[1:t + 1], tokens[t + 1:2 * t + 1]])
    return inputs, targets, len(chars)


@torch.no_grad()
def build():
    inputs, targets, vocab = windows()
    model = with_fixed_weights(Gpt(GptConfig(vocab)), 42).eval()
    counts = [("token embedding", model.tok), ("position embedding", model.pos),
              ("transformer blocks", model.blocks), ("final LayerNorm", model.ln_f),
              ("head", model.head)]
    counts = [(name, sum(p.numel() for p in m.parameters())) for name, m in counts]
    logits = model(inputs)
    n, t, v = logits.shape
    loss = F.cross_entropy(logits.reshape(n * t, v), targets.reshape(n * t)).item()
    changed = inputs.clone()
    changed[:, -1] = (changed[:, -1] + 1) % v
    moved = (model(changed) - logits).abs()
    return dict(counts=counts, num_params=sum(p.numel() for p in model.parameters()),
                logits_dims=list(logits.shape), first=logits[0, 0, :4].tolist(),
                last=logits[1, t - 1, :4].tolist(), loss=loss,
                causal=(moved[:, :-1].max().item(), moved[:, -1].max().item()))


def main():
    r = build()
    print(f"{'component':<20}{'params':>8}")
    for name, count in r["counts"]:
        print(f"{name:<20}{count:>8}")
    print(f"{'parameters()':<20}{r['num_params']:>8}")
    print(f"input [2, 128] -> logits {r['logits_dims']}")
    print("logits[0, 0, :4]   = [" + ", ".join(f"{x:.4f}" for x in r["first"]) + "]")
    print("logits[1, 127, :4] = [" + ", ".join(f"{x:.4f}" for x in r["last"]) + "]")
    print(f"cross-entropy      = {r['loss']:.4f}  (ln 62 = {math.log(62):.4f})")
    print(f"change the last token: positions 0 to 126 move by {r['causal'][0]:.1e}, "
          f"position 127 by {r['causal'][1]:.4f}")


def test_matches_burn():
    r = build()
    assert r["num_params"] == 116224
    assert all(abs(a - b) < 1e-4 for a, b in zip(r["first"], EXPECTED_FIRST))
    assert all(abs(a - b) < 1e-4 for a, b in zip(r["last"], EXPECTED_LAST))
    assert abs(r["loss"] - EXPECTED_LOSS) < 1e-4
    assert r["causal"][0] == 0.0
    assert abs(r["causal"][1] - 0.804828) < 1e-4


EXPECTED_FIRST = [-0.196566, 0.714586, 0.171288, -0.283295]
EXPECTED_LAST = [-0.200070, 1.116861, 0.406603, -0.720139]
EXPECTED_LOSS = 4.362419


if __name__ == "__main__":
    main()
