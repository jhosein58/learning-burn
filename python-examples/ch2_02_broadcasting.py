import torch

m    = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
bias = torch.tensor([100., 200., 300.])

out = m + bias    # PyTorch inserts the missing axis for you
# [[101., 202., 303.], [104., 205., 306.]]
