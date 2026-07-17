"""Example 1.1 - Rank Is Not Shape (PyTorch reference + parity test).

Run the demo:   python python.py
Run the test:   pytest python.py
"""

import torch


def build():
    """A rank-1 tensor that holds five values -- shape [5], not 'rank 5'."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


def main():
    t = build()
    print("tensor =", t)
    print("rank (ndim) =", t.ndim, "  shape =", tuple(t.shape))


def test_matches_burn():
    """Parity with rust.rs: same rank, same shape, same values."""
    t = build()

    # rank 1, shape [5]
    assert t.ndim == 1
    assert tuple(t.shape) == (5,)

    # GOLDEN VALUES -- identical to the assertion in rust.rs
    assert t.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


if __name__ == "__main__":
    main()
