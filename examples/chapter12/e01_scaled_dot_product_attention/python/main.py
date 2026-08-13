import torch

Q = torch.eye(2)                        # identity, for readability
K = torch.eye(2)
V = torch.tensor([[1., 2.], [3., 4.]])

d_k    = Q.shape[1]
scores = Q @ K.T / d_k ** 0.5           # [[0.707, 0], [0, 0.707]]
w      = torch.softmax(scores, dim=1)   # [[0.670, 0.330], [0.330, 0.670]]
out    = w @ V                          # [[1.660, 2.660], [2.340, 3.340]]

print("scores  =")
print(scores)  # [[0.7071, 0.0000], [0.0000, 0.7071]]

print("\nweights =")
print(w)       # [[0.6698, 0.3302], [0.3302, 0.6698]]

print("\noutput  =")
print(out)     # [[1.6605, 2.6605], [2.3395, 3.3395]]
