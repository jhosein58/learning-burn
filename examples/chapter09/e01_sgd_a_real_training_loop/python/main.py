import torch
import torch.nn as nn

model = nn.Linear(1, 1)                     # y = w*x + b
optim = torch.optim.SGD(model.parameters(), lr=0.02)

x = torch.tensor([[1.], [2.], [3.], [4.]])
y = torch.tensor([[3.], [5.], [7.], [9.]])  # y = 2x + 1

for _ in range(3000):
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    optim.zero_grad()
    loss.backward()                         # <-- Burn does not need this
    optim.step()

print(model(torch.tensor([[5.]])))          # ~11
