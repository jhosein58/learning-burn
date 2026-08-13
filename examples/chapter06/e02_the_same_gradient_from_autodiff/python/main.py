import torch

X = torch.tensor([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
y = torch.tensor([[3.], [5.], [7.], [9.]])

w = torch.zeros(2, 1, requires_grad=True)

pred = X @ w
diff = pred - y
loss = (diff * diff).mean()         # MSE

loss.backward()                     # autodiff does the calculus
print(loss)                         # 41.0
print(w.grad)                       # [[-12.], [-35.]]
