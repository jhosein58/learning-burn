"""Example 16.1 - Will It Fit? (Python reference + parity test).

The same budget as rust.rs: count every tensor of the esp32-tinyLLM model
from its config, price it at f32 and at group-128 int4, and compare with the
ESP32-S3 N16R8's SRAM, PSRAM and flash. The int4 total must equal the size
of the real model.bin, byte for byte.

Run:   python main.py
Test:  pytest main.py
"""

V, D, L, F, P, GROUP = 32768, 96, 6, 66, 128, 128
SRAM, PSRAM, FLASH = 512 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024


def ceil_div(a, b):
    return -(-a // b)


def tensors():
    t = [("token embedding", V, D), ("PLE model proj", L * P, D),
         ("PLE proj norm", P, 0), ("PLE table", V, L * P)]
    for _ in range(L):
        t += [("attn norm", D, 0), ("qkv", 3 * D, D), ("attn proj", D, D),
              ("ffn norm", D, 0), ("gate", F, D), ("up", F, D), ("down", D, F),
              ("ple gate", P, D), ("ple proj", D, P), ("ple norm", D, 0)]
    t.append(("out norm", D, 0))
    return t


def params(rows, cols):
    return rows if cols == 0 else rows * cols


def int4_bytes(rows, cols):
    if cols == 0:
        return rows * 4
    return 4 + rows * ceil_div(cols, 2) + rows * ceil_div(cols, GROUP) * 2


def build():
    t = tensors()
    total = sum(params(r, c) for _, r, c in t)
    ple = params(V, L * P)
    file = 40 + sum(int4_bytes(r, c) for _, r, c in t)
    return total, ple, total * 4, file


def main():
    total, ple, f32_bytes, file = build()
    print(f"parameters      = {total}")
    print(f"  in PLE table  = {ple} ({100 * ple / total:.1f}%)")
    print(f"f32 weights     = {f32_bytes / 1e6:.1f} MB")
    print(f"int4 model.bin  = {file} bytes ({file / 1e6:.2f} MB)")


def test_matches_burn():
    total, ple, f32_bytes, file = build()
    assert total == 28_869_920
    assert ple == 25_165_824
    assert f32_bytes == 115_479_680
    assert file == 14_912_332
    assert 25_353 * ceil_div(D, 2) == 1_216_944


if __name__ == "__main__":
    main()
