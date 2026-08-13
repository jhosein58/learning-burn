import torch
import torch.nn.functional as F

x = torch.tensor([-2., -0.5, 0., 1., 3.])

relu  = F.relu(x)              # [0, 0, 0, 1, 3]
leaky = F.leaky_relu(x, 0.1)   # [-0.2, -0.05, 0, 1, 3]
gelu  = F.gelu(x)              # smooth

print(f"relu       = {relu}")
print(f"leaky_relu = {leaky}")
print(f"gelu       = {gelu}")
