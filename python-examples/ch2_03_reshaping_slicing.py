import torch

t     = torch.arange(0, 12)
m     = t.reshape(3, 4)
flat  = m.flatten()        # back to a flat [12]
piece = m[0:2, 1:3]        # rows 0-1, cols 1-2
col   = m.narrow(1, 0, 1)  # first column
rows  = m[[0, 2], :]       # gather rows 0 and 2
