"""Example 1.4 - Tensors Filled With Known Values (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    z = torch.zeros(2, 3)
    o = torch.ones(2, 3)
    f = torch.full((2, 3), 7.0)
    eye = torch.eye(3)
    r = torch.arange(5, 10)  # [5, 6, 7, 8, 9]
    return z, o, f, eye, r


def main():
    z, o, f, eye, r = build()
    for name, t in [("zeros", z), ("ones", o), ("full", f), ("eye", eye), ("range", r)]:
        print(name, "=", t)


def test_matches_burn():
    z, o, f, eye, r = build()
    assert z.flatten().tolist() == [0.0] * 6
    assert o.flatten().tolist() == [1.0] * 6
    assert f.flatten().tolist() == [7.0] * 6
    assert eye.flatten().tolist() == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert r.tolist() == [5, 6, 7, 8, 9]


if __name__ == "__main__":
    main()
