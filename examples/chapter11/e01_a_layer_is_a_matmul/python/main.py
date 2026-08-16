import torch

x = torch.tensor([[1., 2.], [3., 4.]])          # 2 examples, 2 features
t = torch.tensor([[1.], [0.]])                  # one target per example, [2, 1]
W1 = torch.tensor([[.1, .2, .3], [.4, .5, .6]]) # [2, 3]
W2 = torch.tensor([[.7], [.8], [.9]])           # [3, 1]

h_pre = x @ W1                                  # [2, 3]
h = torch.relu(h_pre)                           # [2, 3]
out = h @ W2                                    # [2, 1] -> [[2.94], [6.38]]

# Mean squared error: square the per-example errors, then average to a scalar.
diff = out - t                             # [2, 1]
sq_err = diff * diff                            # [2, 1]
loss = sq_err.mean()                            # scalar

print(f"xshape  = {list(x.shape)}")
print(f"W1shape = {list(W1.shape)}")
print(f"W2shape = {list(W2.shape)}")
print(f"h_pre   = {h_pre}")
print(f"h       = {h}")
print(f"out     = {out}")
print(f"t  = {t}")
print(f"sq_err  = {sq_err}")
print(f"loss    = {loss}")                      # 22.234 (scalar MSE)
