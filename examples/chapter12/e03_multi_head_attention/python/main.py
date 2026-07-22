import torch
import torch.nn as nn

seq     = 3
d_model = 4
n_heads = 2
d_k     = d_model // n_heads  # 2

x = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                   [0.0, 1.0, 0.0, 1.0],
                   [1.0, 1.0, 0.0, 0.0]])

wq = nn.Linear(d_model, d_model)
wk = nn.Linear(d_model, d_model)
wv = nn.Linear(d_model, d_model)
wo = nn.Linear(d_model, d_model)

# 1. project
q = wq(x); k = wk(x); v = wv(x)                     # [seq, d_model]

# 2. split into heads
q = q.reshape(seq, n_heads, d_k).transpose(0, 1)    # [heads, seq, d_k]
k = k.reshape(seq, n_heads, d_k).transpose(0, 1)
v = v.reshape(seq, n_heads, d_k).transpose(0, 1)

# 3. attention, batched over the head axis
scores  = q @ k.transpose(1, 2) / d_k ** 0.5         # [heads, seq, seq]
weights = torch.softmax(scores, dim=-1)
ctx     = weights @ v                                # [heads, seq, d_k]

# 4. glue the heads back together
out = wo(ctx.transpose(0, 1).reshape(seq, d_model))  # [seq, d_model]

print(f"inputshape  = {list(x.shape)}")              # [3, 4]
print(f"scoresshape = {list(weights.shape)}")        # [2, 3, 3]
print(f"outputshape = {list(out.shape)}")            # [3, 4]
print(f"weight row sums (should all be 1) =\n{weights.sum(dim=2).detach()}")
