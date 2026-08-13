import torch
import torch.nn.functional as F

x      = torch.tensor([[1., 2.]])
target = torch.tensor([[1.]])

W1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
W2 = torch.tensor([[0.5], [0.6]],           requires_grad=True)

h    = F.relu(x @ W1)  # [[0.7, 1.0]]
out  = h @ W2          # [[0.95]]
loss = ((out - target) ** 2).sum()

loss.backward()        # fills W1.grad AND W2.grad


print("out  =", out.detach().numpy())    # [[0.95]]
print("loss =", loss.item())             # 0.0025
print("gW2  =\n", W2.grad.numpy())       # [[-0.07], [-0.10]]
print("gW1  =\n", W1.grad.numpy())       # [[-0.05, -0.06], [-0.10, -0.12]]
