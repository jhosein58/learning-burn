import torch
import torch.nn as nn
import os
import tempfile

# --- train ---
model = nn.Linear(1, 1)

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

optim = torch.optim.SGD(model.parameters(), lr=0.02)
loss_fn = nn.MSELoss()

for _ in range(3000):
    optim.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optim.step()

# --- save ---
path = os.path.join(tempfile.gettempdir(), "burn_ch13_model.pt")
torch.save(model.state_dict(), path)
print(f'saved weights to "{path}"')

# --- load into a fresh model, elsewhere ---
infer = nn.Linear(1, 1)
infer.load_state_dict(torch.load(path, weights_only=True))
infer.eval()  # <- Burn: just don't use Autodiff

pred = infer(torch.tensor([[5.]]))
print(f"prediction for x=5 after reload: [{pred.item():.1f}]")  # ~11 (= 2*5 + 1)
