import torch

x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

W1 = torch.rand(2, 8) * 2 - 1; W1.requires_grad_()
b1 = torch.zeros(1, 8); b1.requires_grad_()
W2 = torch.rand(8, 1) * 2 - 1; W2.requires_grad_()
b2 = torch.zeros(1, 1); b2.requires_grad_()

lr = 0.1

for epoch in range(40000):
    h = torch.tanh(x @ W1 + b1)
    out = h @ W2 + b2
    diff = out - y
    loss = (diff * diff).mean()

    if epoch % 8000 == 0:
        print(f"epoch {epoch:>5}  loss = {loss.item():.8f}")

    loss.backward()

    with torch.no_grad():
        W1 -= lr * W1.grad; W1.grad.zero_()
        b1 -= lr * b1.grad; b1.grad.zero_()
        W2 -= lr * W2.grad; W2.grad.zero_()
        b2 -= lr * b2.grad; b2.grad.zero_()

with torch.no_grad():       # <-- Burn: .inner() / from_inner()
    h = torch.tanh(x @ W1 + b1)
    preds = h @ W2 + b2
    print(f"\nfinal predictions (target 0, 1, 1, 0):\n{preds.squeeze().tolist()}")
