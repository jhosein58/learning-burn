"""Example 1.8 - Ownership and .clone() (PyTorch reference + parity test).

The maths is identical to Burn's; only Burn makes each reuse explicit with
.clone(). Here we just check the numbers match.

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return (x - x.min()) / (x.max() - x.min())


def main():
    print(build())


def test_matches_burn():
    got = build().tolist()
    expected = [0.0, 1 / 3, 2 / 3, 1.0]
    assert len(got) == 4
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-6


if __name__ == "__main__":
    main()
