import torch

m = torch.tensor([[1., 2.], [3., 4.]])
v = torch.tensor([1., 1.])

mt = m.T                      # transpose
mv = m @ v                    # matrix @ vector -> [3., 7.]
d  = torch.dot(v, v)          # scalar          -> 2.
op = torch.outer(v, v)        # [2, 2] of ones
