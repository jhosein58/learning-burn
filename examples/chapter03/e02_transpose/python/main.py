import torch

m = torch.tensor([[1., 2.],
                  [3., 4.]])

mt = m.T                # or m.transpose(0, 1)
# tensor([[1., 3.],
#         [2., 4.]])
