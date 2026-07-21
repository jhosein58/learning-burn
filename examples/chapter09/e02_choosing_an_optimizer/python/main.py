import torch

# A simple trainable parameter used only to create the optimizers.
params = [torch.nn.Parameter(torch.randn(1))]

sgd = torch.optim.SGD(params, lr=0.02)
sgd_m = torch.optim.SGD(params, lr=0.02, momentum=0.9)
adam = torch.optim.Adam(params, lr=1e-3)
adamw = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)

print("Optimizers created: SGD, SGD with momentum, Adam, AdamW")
