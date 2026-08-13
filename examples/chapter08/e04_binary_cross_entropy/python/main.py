import torch
import torch.nn.functional as F

logits  = torch.tensor([2.0, -1.5, 0.3])
targets = torch.tensor([1., 0., 1.])
loss    = F.binary_cross_entropy_with_logits(logits, targets)

print("loss =", loss.item())  # ~0.294
