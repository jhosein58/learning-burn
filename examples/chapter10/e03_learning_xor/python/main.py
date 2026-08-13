import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
optim = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y = torch.tensor([[0.],    [1.],    [1.],    [0.]])    # XOR

for _ in range(40000):
    loss = nn.functional.mse_loss(model(x), y)
    optim.zero_grad(); loss.backward(); optim.step()
