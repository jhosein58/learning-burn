import torch

a = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])   # [2, 3]

b = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.]])        # [3, 2]

c = a @ b                           # [2, 2]

print(c)
# tensor([[ 4.,  5.],
#         [10., 11.]])
