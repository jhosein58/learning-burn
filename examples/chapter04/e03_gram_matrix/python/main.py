import torch
X = torch.tensor([[1., 1.],
                  [1., 2.],
                  [1., 3.]])       # [3 samples, 2 features]

y = torch.tensor([[6.], [0.], [0.]])

gram = X.T @ X                     # [[3., 6.], [6., 14.]]
xty  = X.T @ y                     # [[6.], [6.]]

w = torch.linalg.solve(gram, xty)  # the closed-form fit

print(f"X^T X =\n{gram}")
print(f"\nX^T y =\n{xty}")
print(f"\nw =\n{w}")
