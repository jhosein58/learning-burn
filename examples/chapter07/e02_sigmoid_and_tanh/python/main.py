import torch

x = torch.tensor([-1., 0., 1.])

sigmoid = torch.sigmoid(x)  # [0.2689, 0.5, 0.7311]
tanh    = torch.tanh(x)     # [-0.7616, 0.0, 0.7616]

print(f"sigmoid = {sigmoid}")
print(f"tanh    = {tanh}")
