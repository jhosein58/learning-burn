import torch
import torch.nn.functional as F

pred   = torch.tensor([[1., 2.], [3., 4.]])
target = torch.tensor([[2., 1.], [3., 2.]])
loss   = F.huber_loss(pred, target, delta=1.0)

print(f"loss = {loss.item()}")  # 0.625
