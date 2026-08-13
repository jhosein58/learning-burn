import torch

m   = torch.tensor([[1., 2.], [3., 4.]])
eye = torch.eye(3)              # 3x3 identity

lo  = torch.tril(m)             # [[1., 0.], [3., 4.]]
up  = torch.triu(m)             # [[1., 2.], [0., 4.]]
tr  = torch.trace(m)            # 1 + 4 = 5.
