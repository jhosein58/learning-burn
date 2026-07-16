import torch

x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])

mean = x.mean(dim=0, keepdim=True)        # [[3., 4.]]
std  = x.var(dim=0, keepdim=True).sqrt()  # [[2., 2.]]   (unbiased, /(n-1))
z    = (x - mean) / std                   # [[-1,-1], [0,0], [1,1]]
