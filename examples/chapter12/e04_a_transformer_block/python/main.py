import torch
import torch.nn as nn


def layer_norm(x):
    mean = x.mean(dim=1, keepdim=True)
    centered = x - mean
    var = (centered ** 2).mean(dim=1, keepdim=True)
    return centered / (var + 1e-5).sqrt()


d_model = 4
d_ff    = 8

x = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0, 0.0]])

# Self-attention projections.
wq = nn.Linear(d_model, d_model)
wk = nn.Linear(d_model, d_model)
wv = nn.Linear(d_model, d_model)
wo = nn.Linear(d_model, d_model)

# Position-wise feed-forward: d_model -> d_ff -> d_model.
ff1 = nn.Linear(d_model, d_ff)
ff2 = nn.Linear(d_ff, d_model)


def self_attention(x):
    q = wq(x)
    k = wk(x)
    v = wv(x)
    scale = 1.0 / d_model ** 0.5
    scores = q @ k.T * scale                 # [seq, seq]
    weights = torch.softmax(scores, dim=1)
    return wo(weights @ v)                   # [seq, d_model]


def feed_forward(x):
    return ff2(torch.relu(ff1(x)))           # [seq, d_model]


x = layer_norm(x + self_attention(x))        # sub-layer 1
x = layer_norm(x + feed_forward(x))          # sub-layer 2

out = x.detach()

print(f"input  shape = {list(out.shape)}")   # [3, 4]
print(f"output shape = {list(out.shape)}")   # [3, 4]  (a block preserves shape)
print(f"output       =\n{out}")
# After the final LayerNorm, each row is ~zero-mean:
print(f"row means (≈ 0) =\n{out.mean(dim=1)}")
