import torch
import torch.nn.functional as F

pred = torch.tensor([[1., 2.], [3., 4.]])
target = torch.tensor([[2., 1.], [3., 2.]])

loss = F.mse_loss(pred, target, reduction="mean")

# squared errors: [1, 1, 0, 4] -> mean = 1.5
print("MSE =", loss.item())  # MSE = 1.5
