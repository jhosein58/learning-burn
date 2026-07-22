import torch

Q = torch.eye(2)
K = torch.eye(2)
V = torch.tensor([[1., 2.], [3., 4.]])
d_k = Q.shape[1]

scores = Q @ K.T / d_k ** 0.5

mask   = torch.tensor([[0., -1e9],   # token 0 must not see token 1
                       [0.,  0.]])   # token 1 may see both

w      = torch.softmax(scores + mask, dim=1)
out    = w @ V

# row 0 weights -> [1, 0]; output -> V's first row, exactly

print(f"masked weights =\n{w}")     # [[1, 0], [0.3302, 0.6698]]
print(f"output         =\n{out}")   # [[1, 2], [2.3395, 3.3395]]
