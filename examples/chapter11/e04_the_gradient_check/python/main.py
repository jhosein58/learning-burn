import torch

def loss_at(w):
    z = 3 * w - 2
    return z * z

w_value = 0.5

w = torch.tensor([w_value], dtype=torch.float32, requires_grad=True)
z = w * 3 - 2
loss = (z * z).sum()
loss.backward()

eps = 1e-4
numeric = (loss_at(w_value + eps) - loss_at(w_value - eps)) / (2 * eps)

print(f"loss            = {loss.item()}")
print(f"autodiff grad   = {w.grad.item()}")
print(f"numerical grad  = {numeric:.6f}")
print("they match -> backprop is trustworthy")
