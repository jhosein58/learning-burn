"""Example 15.3 - Pooling and the Receptive Field (PyTorch reference + parity test).

Max, average and global-average pooling on a 4x4 image of 1..16, and the
receptive field of a conv -> pool -> conv stack.

Run:   python main.py
Test:  pytest main.py
"""

import torch
import torch.nn as nn


def receptive_fields(layers):
    rf, jump, out = 1, 1, []
    for name, k, stride in layers:
        rf += (k - 1) * jump
        jump *= stride
        out.append((name, rf))
    return out


def build():
    x = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    mx = nn.MaxPool2d(2)(x)
    avg = nn.AvgPool2d(2)(x)
    glob = nn.AdaptiveAvgPool2d(1)(x)
    rf = receptive_fields([("conv 3x3", 3, 1), ("pool 2x2", 2, 2), ("conv 3x3", 3, 1)])
    return mx.flatten().tolist(), avg.flatten().tolist(), glob.flatten().tolist(), rf


def main():
    mx, avg, glob, rf = build()
    print("max pool 2x2    =", mx)
    print("avg pool 2x2    =", avg)
    print("global avg pool =", glob)
    for name, size in rf:
        print(f"after {name}: each output sees {size}x{size} input pixels")


def test_matches_burn():
    mx, avg, glob, rf = build()
    assert mx == [6.0, 8.0, 14.0, 16.0]
    assert avg == [3.5, 5.5, 11.5, 13.5]
    assert glob == [8.5]
    assert [r[1] for r in rf] == [3, 4, 8]


if __name__ == "__main__":
    main()
