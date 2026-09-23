"""Example 17.1 - A Dataset from a CSV File (PyTorch reference + parity test).

Read machines.csv, standardise the three features with statistics computed
from the data, and batch it with a DataLoader: 120 rows in batches of 32.

Run:   python main.py
Test:  pytest main.py
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "machines.csv")


class Machines(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.rows = [(float(r["temperature"]), float(r["vibration"]), float(r["rpm"]),
                          int(r["failure"])) for r in csv.DictReader(f)]
        x = np.array([r[:3] for r in self.rows], dtype=np.float64)
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)   # population std, like rust.rs

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        *x, y = self.rows[i]
        x = (np.array(x) - self.mean) / self.std
        return torch.tensor(x, dtype=torch.float32), y


def build():
    data = Machines(CSV)
    print("first row  =", data.rows[0])
    loader = DataLoader(data, batch_size=32, shuffle=False)
    sizes, first = [], None
    for features, targets in loader:
        if first is None:
            print("batch      = features", list(features.shape), ", targets", list(targets.shape))
            first = features[0].tolist()
        sizes.append(len(targets))
    failures = sum(r[3] for r in data.rows)
    return len(data), failures, data.mean.tolist(), data.std.tolist(), sizes, first


def main():
    rows, failures, mean, std, sizes, first = build()
    print(f"rows       = {rows} ({failures} failures)")
    print("mean       =", [round(v, 3) for v in mean])
    print("std        =", [round(v, 3) for v in std])
    print("batches    =", sizes)
    print("row 0, standardised =", [round(v, 4) for v in first])


def test_matches_burn():
    rows, failures, mean, std, sizes, first = build()
    assert (rows, failures) == (120, 42)
    assert sizes == [32, 32, 32, 24]
    assert all(abs(a - b) / abs(b) < 1e-5 for a, b in zip(mean, EXPECTED_MEAN))
    assert all(abs(a - b) / abs(b) < 1e-5 for a, b in zip(std, EXPECTED_STD))
    assert all(abs(a - b) < 1e-4 for a, b in zip(first, EXPECTED_ROW0))


EXPECTED_MEAN = [66.823333, 5.334917, 2198.6083]
EXPECTED_STD = [14.729815, 2.487775, 793.63241]
EXPECTED_ROW0 = [-1.488364, -0.625023, -1.367898]


if __name__ == "__main__":
    main()
