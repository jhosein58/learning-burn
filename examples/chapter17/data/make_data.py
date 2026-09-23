"""Regenerates this folder's data files. Deterministic: the same bytes every run.

  machines.csv     120 readings from imaginary machines: temperature (C),
                   vibration (mm/s), rpm, and whether the machine failed
                   within the next day (1) or not (0).
  shapes/<class>/  24 small grayscale PNGs (12x12), 6 per class, drawn by the
                   same generator as Chapter 15's examples.

Run:  python make_data.py   (needs numpy + Pillow)
"""
import csv
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


class Lcg:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.next() / 32767


def machines():
    rng = Lcg(2026)
    rows = []
    for _ in range(120):
        temp = round(rng.uniform(40.0, 95.0), 1)
        vib = round(rng.uniform(0.5, 9.5), 2)
        rpm = int(rng.uniform(900, 3600))
        risk = 0.04 * (temp - 70) + 0.35 * (vib - 5) + 0.0006 * (rpm - 2200)
        failure = 1 if risk + rng.uniform(-0.6, 0.6) > 0.5 else 0
        rows.append((temp, vib, rpm, failure))
    with open(os.path.join(HERE, "machines.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature", "vibration", "rpm", "failure"])
        w.writerows(rows)


def shapes():
    SIZE = 12
    names = ["0_horizontal", "1_vertical", "2_diagonal", "3_plus"]
    rng = Lcg(99)
    for i in range(24):
        cls = i % 4
        img = np.zeros((SIZE, SIZE), dtype=np.float32)
        if cls == 0:
            r, c = rng.next() % 12, rng.next() % 7
            img[r, c:c + 6] = 1.0
        elif cls == 1:
            c, r = rng.next() % 12, rng.next() % 7
            img[r:r + 6, c] = 1.0
        elif cls == 2:
            r, c = rng.next() % 7, rng.next() % 7
            for k in range(6):
                img[r + k, c + k] = 1.0
        else:
            r, c = 2 + rng.next() % 8, 2 + rng.next() % 8
            img[r, c - 2:c + 3] = 1.0
            img[r - 2:r + 3, c] = 1.0
        noise = np.array([rng.next() for _ in range(SIZE * SIZE)]).reshape(SIZE, SIZE)
        pixels = np.clip(img * 215 + noise % 40, 0, 255).astype(np.uint8)
        folder = os.path.join(HERE, "shapes", names[cls])
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(pixels, mode="L").save(os.path.join(folder, f"{i // 4:02d}.png"))


if __name__ == "__main__":
    machines()
    shapes()
    print("wrote machines.csv and shapes/")
