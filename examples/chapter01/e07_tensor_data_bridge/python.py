"""Example 1.7 - Getting Your Numbers Back Out (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    return torch.tensor([1.0, 2.0, 3.0])


def main():
    t = build()
    # PyTorch's bridge back out: .tolist() copies to Python; .numpy() shares.
    print("values =", t.tolist())


def test_matches_burn():
    assert build().tolist() == [1.0, 2.0, 3.0]


if __name__ == "__main__":
    main()
