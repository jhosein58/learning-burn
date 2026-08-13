import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([4., 5.], requires_grad=True)

loss = (a * b).sum()    # 2*4 + 3*5 = 23
loss.backward()         # returns nothing; mutates a and b

# the gradients are now attributes hanging ON the tensors
print(a.grad)           # tensor([4., 5.])  == b
print(b.grad)           # tensor([2., 3.])  == a
