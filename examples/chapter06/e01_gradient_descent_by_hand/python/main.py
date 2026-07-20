import torch

X = torch.tensor([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
y = torch.tensor([[3.], [5.], [7.], [9.]])

w = torch.zeros(2, 1)
n, lr = 4.0, 0.05

for _ in range(3000):
    err  = X @ w - y               # Xw - y
    grad = (2 / n) * X.T @ err     # the hand-derived gradient
    w    = w - lr * grad           # downhill

print(w)                           # ~[[1.], [2.]]  (bias, slope)
