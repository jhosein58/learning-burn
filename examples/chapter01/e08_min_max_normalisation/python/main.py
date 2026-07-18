import torch

input = torch.tensor([1., 2., 3., 4.])

mn = input.min()
mx = input.max()
input = (input - mn) / (mx - mn)

print(input)    # [0.0000, 0.3333, 0.6667, 1.0000]
