import torch

x = torch.tensor([[1., 2.], [3., 4.]])

total    = x.sum()                        # 10.
col_sum  = x.sum(dim=0, keepdim=True)     # [[4., 6.]]
row_mean = x.mean(dim=1, keepdim=True)    # [[1.5], [3.5]]
m, idx   = x.max(dim=1)                   # values [2., 4.], indices [1, 1]
