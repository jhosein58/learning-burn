import torch

a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

u    = a * b     # 6
v    = u * u     # 36
loss = v.sum()   # 36

loss.backward()

print(a.grad)    # tensor([36.])  == 2u*b
print(b.grad)    # tensor([24.])  == 2u*a
