"""Example 15.1 - 1D Convolution by Hand (PyTorch reference + parity test).

A kernel [1, 0, -1] slid along a signal. Done with a loop, then with
F.conv1d. Both give the same four numbers as Burn.

Run:   python main.py
Test:  pytest main.py
"""

import torch
import torch.nn.functional as F


def by_hand(signal, kernel):
    k = len(kernel)
    return [sum(signal[i + j] * kernel[j] for j in range(k))
            for i in range(len(signal) - k + 1)]


def build():
    signal = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
    kernel = [1.0, 0.0, -1.0]
    manual = by_hand(signal, kernel)

    x = torch.tensor(signal).reshape(1, 1, 6)   # [batch, channels, length]
    w = torch.tensor(kernel).reshape(1, 1, 3)   # [out_ch, in_ch, kernel_len]
    y = F.conv1d(x, w)                          # no padding, stride 1
    print("torch shape =", list(y.shape))       # [1, 1, 4]
    return manual, y.flatten().tolist()


def main():
    manual, torch_out = build()
    print("by hand     =", manual)
    print("torch conv1d=", torch_out)


def test_matches_burn():
    manual, torch_out = build()
    assert manual == [-3.0, -5.0, -7.0, -9.0]
    assert torch_out == manual


if __name__ == "__main__":
    main()
