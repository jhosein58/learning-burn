import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
f = (x * x).sum()          # 1 + 4 + 9 = 14

f.backward()               # fills in x.grad, returns nothing

print(f)                   # tensor(14., grad_fn=<SumBackward0>)
print(x.grad)              # tensor([2., 4., 6.])    == 2x
