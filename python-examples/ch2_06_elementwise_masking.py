import torch

x = torch.tensor([-1., 0., 2.])

y      = torch.log1p(torch.exp(x))   # [0.3133, 0.6931, 2.1269]   (softplus)
c      = torch.clamp(x, 0., 1.)      # [0., 0., 1.]
mask   = x > 0                       # [False, False, True]
picked = torch.where(mask, 9., x)    # [-1., 0., 9.]
