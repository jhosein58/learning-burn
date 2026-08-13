import torch
import torch.nn.functional as F

# raw logits — NOT softmaxed
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3]])

targets = torch.tensor([0, 1])        # correct class per row

loss = F.cross_entropy(logits, targets)
print("cross-entropy =", loss.item()) # ~0.319
