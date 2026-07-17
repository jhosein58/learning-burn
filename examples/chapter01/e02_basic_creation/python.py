"""Example 1.2 - Creating Tensors from Data (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build_a():
    return torch.tensor([1.0, 2.0, 3.0])


def build_m():
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def main():
    print("a =", build_a())
    print("m =", build_m())


def test_matches_burn():
    assert build_a().tolist() == [1.0, 2.0, 3.0]

    m = build_m()
    assert tuple(m.shape) == (2, 3)
    assert m.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


if __name__ == "__main__":
    main()
