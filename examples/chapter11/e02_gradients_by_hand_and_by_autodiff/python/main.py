import torch

torch.set_printoptions(precision=7, sci_mode=False)

dtype = torch.float32

x = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]], dtype=dtype)

t = torch.tensor([[1.0],
                  [0.0]], dtype=dtype)

n = x.shape[0]

w1 = torch.tensor([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]], dtype=dtype, requires_grad=True)

w2 = torch.tensor([[0.7],
                   [0.8],
                   [0.9]], dtype=dtype, requires_grad=True)

# ---- forward ----
h_pre = x @ w1
h     = torch.relu(h_pre)
out   = h @ w2
diff  = out - t
loss  = (diff * diff).mean()

# ---- (1) gradients BY HAND ----
dout       = diff * (2.0 / n)
manual_gw2 = h.T @ dout
dh         = dout @ w2.T
dh_pre     = dh * (h_pre > 0).to(dtype)
manual_gw1 = x.T @ dh_pre

# ---- (2) gradients from AUTODIFF ----
loss.backward()
auto_gw1 = w1.grad
auto_gw2 = w2.grad

print(f"loss = {loss.detach().reshape(-1).numpy()}")

print("\n-- gradient for W2 --")
print(f"manual =\n{manual_gw2.detach().reshape(-1).numpy()}")
print(f"auto   =\n{auto_gw2.detach().reshape(-1).numpy()}")

print("\n-- gradient for W1 --")
print(f"manual =\n{manual_gw1.detach().reshape(-1).numpy()}")
print(f"auto   =\n{auto_gw1.detach().reshape(-1).numpy()}")
