"""Example 17.3 - Splitting and Shuffling (PyTorch reference + guarantee test).

A seeded 16/4 train/validation split, and a DataLoader that reshuffles the
training set every epoch. PyTorch's random numbers are not Burn's, so the
orders differ from rust.rs; the test checks the same guarantees instead.

Run:   python main.py
Test:  pytest main.py
"""

import torch
from torch.utils.data import DataLoader, random_split


def build():
    train, val = random_split(range(20), [16, 4], generator=torch.Generator().manual_seed(42))
    train_ids, val_ids = list(train), list(val)

    def make_loader():
        return DataLoader(train_ids, batch_size=4, shuffle=True,
                          generator=torch.Generator().manual_seed(7))

    loader = make_loader()
    epoch1 = [int(i) for b in loader for i in b]
    epoch2 = [int(i) for b in loader for i in b]
    again = [int(i) for b in make_loader() for i in b]
    return train_ids, val_ids, epoch1, epoch2, again


def main():
    train, val, epoch1, epoch2, again = build()
    print(f"train ({len(train)}) =", train)
    print(f"val   ({len(val)})  =", val)
    print("epoch 1    =", epoch1)
    print("epoch 2    =", epoch2)
    print("epoch 1 of a rebuilt loader, same seed =", "identical" if again == epoch1 else "DIFFERENT")
    print("each epoch sees every training item exactly once:",
          sorted(epoch1) == sorted(train) and sorted(epoch2) == sorted(train))


def test_same_guarantees_as_burn():
    train, val, epoch1, epoch2, again = build()
    assert (len(train), len(val)) == (16, 4)
    assert sorted(train + val) == list(range(20))
    assert sorted(epoch1) == sorted(train) and sorted(epoch2) == sorted(train)
    assert epoch1 != epoch2
    assert epoch1 == again


if __name__ == "__main__":
    main()
