import torch

x = torch.tensor([3., 4.])

l2   = torch.linalg.norm(x)                    # 5. — the usual one
l1   = torch.linalg.norm(x, ord=1)             # 7.
linf = torch.linalg.norm(x, ord=float('inf'))  # 4.

print(f"L2 norm: {l2.item()}")
print(f"L1 norm: {l1.item()}")
print(f"L∞ norm: {linf.item()}")
