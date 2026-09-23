# Chapter 17 data

Small, deterministic data files for the Chapter 17 examples. `make_data.py`
regenerates them byte for byte (needs numpy and Pillow).

| Path | What it is |
|---|---|
| `machines.csv` | 120 synthetic machine readings: temperature, vibration, rpm, and a 0/1 failure label |
| `shapes/<class>/NN.png` | 24 grayscale 12x12 PNGs, 6 per class, from the same generator as Chapter 15 |
