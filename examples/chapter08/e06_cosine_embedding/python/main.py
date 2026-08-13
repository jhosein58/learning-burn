import torch
import torch.nn.functional as F

a = torch.tensor([[1., 0.], [1., 0.]])
b = torch.tensor([[0.9, 0.1], [-1., 0.]])
y = torch.tensor([1, -1])

loss = F.cosine_embedding_loss(a, b, y)

print(f"loss = {loss.item()}")  # ~0.003
