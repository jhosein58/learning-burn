import torch

x    = torch.tensor([3., 4.])
n2   = torch.linalg.norm(x)                # 5.0
unit = x / n2                              # [0.6, 0.8]

m    = torch.tensor([[4., 3.], [6., 3.]])
d    = torch.linalg.det(m)                 # -6.0

print(f"n2   = {n2.item()}")
print(f"d    = {d.item()}")
print(f"unit = {unit}")
