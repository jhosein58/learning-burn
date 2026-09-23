"""Example 16.2 - The int4 Dot Product (NumPy reference + parity test).

Pack eight weights in -8..=7 as value+8 nibbles, two per byte, then take
their dot product with i16 activations two ways: subtracting 8 per weight,
and subtracting 8 * sum(activations) once. Same integer, as in rust.rs.

Run:   python main.py
Test:  pytest main.py
"""

import numpy as np

WEIGHTS = np.array([3, -2, 7, -8, 0, 5, -1, 4], dtype=np.int8)
ACTS = np.array([10, -3, 25, 7, -12, 4, 9, -6], dtype=np.int16)


def pack(w):
    n = (w.astype(np.int16) + 8).astype(np.uint8)
    return (n[0::2] | (n[1::2] << 4)).astype(np.uint8)


def build():
    codes = pack(WEIGHTS)
    nibbles = np.empty(8, dtype=np.int32)
    nibbles[0::2] = codes & 0xF
    nibbles[1::2] = codes >> 4
    a = ACTS.astype(np.int32)
    naive = int(((nibbles - 8) * a).sum())
    hoisted = int((nibbles * a).sum() - 8 * a.sum())
    real = np.float32(hoisted) * np.float32(0.05) * np.float32(0.02)
    return codes.tolist(), naive, hoisted, float(real)


def main():
    codes, naive, hoisted, real = build()
    print(f"8 weights -> {len(codes)} bytes:", " ".join(f"0x{b:02X}" for b in codes))
    print("naive dot   =", naive)
    print("hoisted dot =", hoisted)
    print("as a float  =", real)


def test_matches_burn():
    codes, naive, hoisted, real = build()
    assert codes == [0x6B, 0x0F, 0xD8, 0xC7]
    assert naive == 142
    assert hoisted == naive
    assert abs(real - 0.142) < 1e-6


if __name__ == "__main__":
    main()
