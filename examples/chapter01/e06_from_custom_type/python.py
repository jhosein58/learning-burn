"""Example 1.6 - A Tensor From Your Own Struct (PyTorch reference + parity test).

Run:   python python.py
Test:  pytest python.py
"""

from dataclasses import dataclass

import torch


@dataclass
class BodyMetrics:
    age: int
    height: int
    weight: float


def build(bmi):
    return torch.tensor([float(bmi.age), float(bmi.height), bmi.weight])


def main():
    print("t =", build(BodyMetrics(age=25, height=180, weight=80.0)))


def test_matches_burn():
    values = build(BodyMetrics(age=25, height=180, weight=80.0)).tolist()
    assert values == [25.0, 180.0, 80.0]


if __name__ == "__main__":
    main()
