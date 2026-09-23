"""Example 15.2 - 2D Convolution and the Shape Rule (PyTorch reference + parity test).

A vertical-edge kernel on a 5x5 image, then the output-size rule checked
against nn.Conv2d for padding, stride and dilation.

Run:   python main.py
Test:  pytest main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def out_len(n, k, padding, stride, dilation):
    return (n + 2 * padding - dilation * (k - 1) - 1) // stride + 1


def build():
    image = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]] * 5).reshape(1, 1, 5, 5)
    kernel = torch.tensor([[-1.0, 0.0, 1.0]] * 3).reshape(1, 1, 3, 3)
    edges = F.conv2d(image, kernel)
    print("edge map shape =", list(edges.shape))   # [1, 1, 3, 3]

    x = torch.zeros(1, 1, 12, 12)
    cases = [("valid, stride 1", 0, 1, 1), ("padding 1", 1, 1, 1),
             ("padding 1, stride 2", 1, 2, 1), ("dilation 2", 0, 1, 2)]
    shapes = []
    for name, p, s, d in cases:
        conv = nn.Conv2d(1, 8, 3, padding=p, stride=s, dilation=d)
        weight_dims = list(conv.weight.shape)
        shapes.append((name, out_len(12, 3, p, s, d), conv(x).shape[2]))
    return edges.flatten().tolist(), shapes, weight_dims


def main():
    edges, shapes, weight_dims = build()
    print("edge map       =", edges)
    print("conv weight    =", weight_dims, " (out, in, kh, kw)")
    for name, predicted, actual in shapes:
        print(f"12x12, 3x3, {name:<20} -> rule {predicted:>2}, Conv2d {actual:>2}")


def test_matches_burn():
    edges, shapes, weight_dims = build()
    assert edges == [3.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0, 0.0]
    assert weight_dims == [8, 1, 3, 3]
    assert [s[2] for s in shapes] == [10, 12, 6, 8]
    assert all(s[1] == s[2] for s in shapes)


if __name__ == "__main__":
    main()
