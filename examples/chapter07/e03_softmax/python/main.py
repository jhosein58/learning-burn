import torch
import torch.nn.functional as F

logits = torch.tensor([[1., 2., 3.]])  # [batch, classes]

probs = F.softmax(logits, dim=1)
# [[0.0900, 0.2447, 0.6652]] — sums to 1

print(probs.sum())  # tensor(1.)
