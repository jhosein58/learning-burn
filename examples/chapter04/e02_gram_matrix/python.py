"""Example 4.2 - Gram Matrix (PyTorch reference + parity test).

The normal-equation building blocks: X^T X and X^T y.

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    x = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = torch.tensor([[6.0], [0.0], [0.0]])

    xt = x.t()
    gram = xt @ x    # X^T X
    xty = xt @ y     # X^T y
    return gram, xty


def main():
    gram, xty = build()
    print("X^T X =\n", gram)
    print("X^T y =\n", xty)


def test_matches_burn():
    gram, xty = build()

    assert tuple(gram.shape) == (2, 2)
    assert gram.flatten().tolist() == [3.0, 6.0, 6.0, 14.0]

    assert tuple(xty.shape) == (2, 1)
    assert xty.flatten().tolist() == [6.0, 6.0]


if __name__ == "__main__":
    main()
