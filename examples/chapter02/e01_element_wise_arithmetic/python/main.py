import torch

a   = torch.tensor([1., 2., 3.])
b   = torch.tensor([10., 20., 30.])

s   = a + b             # [11., 22., 33.]
p   = a * b             # [10., 40., 90.]    element-wise!
sc  = a * 2.0           # [2., 4., 6.]
neg = -a                # [-1., -2., -3.]
