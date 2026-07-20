import torch

x = torch.tensor([1., 2., 3.])

with torch.no_grad():   # PyTorch: a context manager
    y = x + 5.0         # no graph is recorded inside here

# or, equivalently:
y = x.detach() + 5.0

print(y)
