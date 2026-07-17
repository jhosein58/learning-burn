"""Example 4.1 - Vector Norm, Normalize, Determinant (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    x = torch.tensor([3.0, 4.0])
    n2 = torch.linalg.norm(x)          # 5.0
    unit = x / n2                       # [0.6, 0.8]

    m = torch.tensor([[4.0, 3.0], [6.0, 3.0]])
    d = torch.linalg.det(m)             # -6.0
    return n2, unit, d


def main():
    n2, unit, d = build()
    print("n2   =", n2)
    print("unit =", unit)
    print("det  =", d)


def test_matches_burn():
    n2, unit, d = build()
    assert abs(n2.item() - 5.0) < 1e-6
    u = unit.tolist()
    assert abs(u[0] - 0.6) < 1e-6 and abs(u[1] - 0.8) < 1e-6
    assert abs(d.item() - (-6.0)) < 1e-5


if __name__ == "__main__":
    main()
