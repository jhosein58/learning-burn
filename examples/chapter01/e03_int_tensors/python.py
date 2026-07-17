"""Example 1.3 - Integer Tensors (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    # Integer literals -> torch's default int64, matching Burn's NdArray Int (i64).
    return torch.tensor([1, 2, 3])


def main():
    t = build()
    print("ints =", t, " dtype =", t.dtype)


def test_matches_burn():
    t = build()
    assert tuple(t.shape) == (3,)
    assert str(t.dtype) == "torch.int64"
    assert t.tolist() == [1, 2, 3]


if __name__ == "__main__":
    main()
