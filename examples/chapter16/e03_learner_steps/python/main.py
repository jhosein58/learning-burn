"""Example 16.3 - Sixteen training steps, checked (PyTorch reference + parity test).

The loop Burn's Learner runs in rust/src/main.rs, written out: the first 128 training
windows in order, the LCG starting weights, AdamW and the warm-up + cosine schedule.
Nothing is random, so every step's learning rate and loss must agree with Burn's.

Run:   python main.py
Test:  pytest main.py
"""

import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                    "shakespeare.txt")


# --- the data: Example 16.1 ---
class TextWindows(Dataset):
    def __init__(self, tokens, block, stride):
        self.tokens, self.block, self.stride = tokens, block, stride

    def __len__(self):
        return (len(self.tokens) - self.block - 1) // self.stride + 1

    def __getitem__(self, i):
        start = i * self.stride
        w = self.tokens[start:start + self.block + 1]
        return w[:-1], w[1:]


def corpus():
    text = open(DATA).read()
    chars = sorted(set(text))
    lookup = {c: i for i, c in enumerate(chars)}
    tokens = torch.tensor([lookup[c] for c in text])
    cut = len(tokens) * 9 // 10
    return tokens[:cut], tokens[cut:], chars


# --- the model: Example 16.2 ---
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
        causal = torch.ones(t, t, dtype=torch.bool).triu(1)
        x = self.blocks(x, mask=causal, is_causal=True)
        return self.head(self.ln_f(x))


class Lcg:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

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


# --- the run ---
@dataclass
class TrainingConfig:
    model: GptConfig
    optimizer: dict = field(default_factory=lambda: {"eps": 1e-8, "weight_decay": 0.01})
    num_epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 1e-2
    min_learning_rate: float = 1e-3
    warmup_steps: int = 50
    seed: int = 16


def schedule(cfg, k, total):
    """The rate of step k = 1, ..., total: Burn's Linear, then its Cosine."""
    w, peak, low = cfg.warmup_steps, cfg.learning_rate, cfg.min_learning_rate
    if k <= w:
        return peak * k / w
    cos = math.cos(math.pi * (k - w - 1) / (total - w))
    return low + 0.5 * (peak - low) * (1 + cos)


def run(cfg, train, valid, epochs, shuffle=True):
    """Trains; returns per epoch (train loss, valid loss, valid ppl, valid acc %), the model,
    and the per-step (lr, loss)."""
    model = with_fixed_weights(Gpt(cfg.model), 42)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, **cfg.optimizer)
    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=shuffle,
                        generator=torch.Generator().manual_seed(cfg.seed))
    total = cfg.num_epochs * len(loader)
    history, steps, k = [], [], 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for inputs, targets in loader:
            k += 1
            for group in optim.param_groups:
                group["lr"] = schedule(cfg, k, total)
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            steps.append((schedule(cfg, k, total), loss.item()))
        model.eval()
        with torch.no_grad():
            v_losses, nll, correct, count = [], 0.0, 0, 0
            for inputs, targets in DataLoader(valid, batch_size=cfg.batch_size):
                logits = model(inputs).flatten(0, 1)
                loss = F.cross_entropy(logits, targets.flatten())
                v_losses.append(loss.item())
                nll += loss.item() * targets.numel()
                correct += (logits.argmax(1) == targets.flatten()).sum().item()
                count += targets.numel()
        history.append((sum(losses) / len(losses), sum(v_losses) / len(v_losses),
                        math.exp(nll / count), 100.0 * correct / count))
    return history, model, steps


def build():
    train_ids, valid_ids, chars = corpus()
    cfg = TrainingConfig(GptConfig(len(chars)), num_epochs=2, warmup_steps=4)
    t, n = cfg.model.block_size, cfg.batch_size
    train = TextWindows(train_ids[:8 * n * t + 1], t, t)          # 128 windows, 8 batches
    history, _, steps = run(cfg, train, TextWindows(valid_ids, t, t), cfg.num_epochs,
                            shuffle=False)
    return steps, [h[1] for h in history]


def main():
    steps, valid = build()
    print("step  learning rate  train loss")
    for k, (lr, loss) in enumerate(steps, 1):
        print(f"{k:>4}  {lr:>13.6f}  {loss:>10.4f}")
    print(f"valid loss after epochs 1 and 2: {valid[0]:.4f}, {valid[1]:.4f}")


def test_matches_burn():
    steps, valid = build()
    assert len(steps) == 16
    assert all(abs(lr - e) < 1e-9 for (lr, _), e in zip(steps, EXPECTED_LR))
    assert all(abs(loss - e) < 1e-3 for (_, loss), e in zip(steps, EXPECTED_LOSS))
    assert abs(valid[1] - EXPECTED_VALID) < 1e-3   # Burn's valid loss after epoch 2


EXPECTED_LR = [0.0025, 0.005, 0.0075, 0.01, 0.01, 0.0098466662, 0.0093971143, 0.0086819805,
               0.00775, 0.0066646857, 0.0055, 0.0043353143, 0.00325, 0.0023180195,
               0.0016028857, 0.0011533338]
EXPECTED_LOSS = [4.382179, 3.749729, 3.523268, 3.410230, 3.407179, 3.451118, 3.302796,
                 3.264419, 3.217106, 3.204884, 3.246782, 3.218565, 3.262942, 3.351518,
                 3.195130, 3.148308]
EXPECTED_VALID = 3.265006


if __name__ == "__main__":
    main()
