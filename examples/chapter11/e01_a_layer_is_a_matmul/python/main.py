import torch

x = torch.tensor([[1., 2.], [3., 4.]])          # 2 examples, 2 features
W1 = torch.tensor([[.1, .2, .3], [.4, .5, .6]]) # [2, 3]
W2 = torch.tensor([[.7], [.8], [.9]])           # [3, 1]

h_pre = x @ W1                                  # [2, 3]
h = torch.relu(h_pre)                           # [2, 3]
out = h @ W2                                    # [2, 1] -> [[2.94], [6.38]]

print(f"xshape  = {list(x.shape)}")
print(f"W1shape = {list(W1.shape)}")
print(f"W2shape = {list(W2.shape)}")
print(f"h_pre   =\n{h_pre}")
print(f"h       =\n{h}")
print(f"out     =\n{out}")
