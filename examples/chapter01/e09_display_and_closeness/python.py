"""Example 1.9 - Display and Closeness (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

import torch


def build_full():
    return torch.full((2, 3), 0.123456789)  # float32


def max_abs_diff():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.001])
    return (a - b).abs().max().item()


def main():
    t = build_full()
    print(t)
    torch.set_printoptions(precision=2)
    print(t)
    torch.set_printoptions(precision=4)

    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.001])
    # never compare floats with ==; compare with a tolerance
    print("allclose(atol=1e-2):", torch.allclose(a, b, atol=1e-2))
    print("allclose(atol=1e-4):", torch.allclose(a, b, atol=1e-4))


def test_matches_burn():
    values = build_full().flatten().tolist()
    assert len(values) == 6
    for x in values:
        assert abs(x - 0.12345679) < 1e-6

    d = max_abs_diff()
    assert 1e-4 < d < 1e-2


if __name__ == "__main__":
    main()
