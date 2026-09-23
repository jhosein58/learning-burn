"""Example 17.5 - Keeping the Model Fed: Worker Processes (PyTorch reference).

The same deliberately slow dataset (4 ms per item) loaded with 0, 2 and 4
workers. Timings are machine-dependent; the test checks that every item
arrives exactly once, as rust.rs does.

Run:   python main.py
Test:  pytest main.py
"""

import time

from torch.utils.data import DataLoader, Dataset


class SlowDataset(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        time.sleep(0.004)   # stands in for opening a file and decoding an image
        return i


def one_epoch(workers):
    loader = DataLoader(SlowDataset(), batch_size=8, num_workers=workers)
    start = time.perf_counter()
    seen = sorted(int(i) for batch in loader for i in batch)
    return seen, time.perf_counter() - start


def build():
    return [(w, *one_epoch(w)) for w in (0, 2, 4)]


def main():
    runs = build()
    base = runs[0][2]
    for workers, seen, t in runs:
        label = "no workers" if workers == 0 else f"{workers} workers"
        print(f"{label:<10}: {len(seen)} items in {t * 1000:>4.0f} ms  "
              f"({len(seen) / t:>5.0f} items/s, {base / t:.1f}x)")


def test_same_guarantees_as_burn():
    for _, seen, _ in build():
        assert seen == list(range(64))


if __name__ == "__main__":
    main()
