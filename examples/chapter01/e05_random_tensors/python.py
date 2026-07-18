"""Example 1.5 - Random Tensors (PyTorch reference + parity test).

Random draws never match Burn element-for-element (different RNGs). Parity is
enforced on the DISTRIBUTION: same contract, large sample, loose tolerance.

Run:   python python.py
Test:  pytest python.py
"""

import torch

N = 100_000


def uniform(n):
    return torch.rand(n)  # uniform on [0, 1)


def normal(n):
    return torch.randn(n)  # standard normal, mean 0 std 1


def main():
    print("uniform =", torch.rand(2, 3))
    print("normal =", torch.randn(2, 3))


def test_uniform_contract():
    t = uniform(N)
    assert t.min().item() >= 0.0
    assert t.max().item() < 1.0
    assert abs(t.mean().item() - 0.5) < 0.05


def test_normal_contract():
    t = normal(N)
    assert abs(t.mean().item() - 0.0) < 0.05
    assert abs(t.std().item() - 1.0) < 0.05


if __name__ == "__main__":
    main()
