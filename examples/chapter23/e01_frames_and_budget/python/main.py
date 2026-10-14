"""Example 23.1 - Frames, hops and the price of forgetting (Python reference + test).

The same budget as rust/src/main.rs: frames in a clip, the receptive field of
four dilated causal convs, multiply-adds per hop for streaming and for two
ways of recomputing, and the floats each approach keeps. PyTorch then confirms
the receptive field: the gradient of the newest output reaches exactly 31
input frames.

Run:   python main.py
Test:  pytest main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE, FRAME, HOP, CLIP = 16_000, 400, 160, 16_000
FEATURES, CHANNELS, K, DILATIONS = 40, 64, 3, [1, 2, 4, 8]
CLOCK_HZ = 240_000_000


def ms(samples):
    return samples * 1000 / SAMPLE_RATE


def layers():
    return [(FEATURES if l == 0 else CHANNELS, CHANNELS, d) for l, d in enumerate(DILATIONS)]


def build():
    frames = 1 + (CLIP - FRAME) // HOP
    ls = layers()
    receptive = 1 + sum(d * (K - 1) for _, _, d in ls)
    macs = [co * ci * K for ci, co, _ in ls]
    macs_stream = sum(macs)
    need, macs_cone = 1, 0
    for l in reversed(range(len(ls))):
        macs_cone += need * macs[l]
        need += ls[l][2] * (K - 1)
    macs_window = frames * macs_stream
    state_floats = sum(d * (K - 1) * ci for ci, _, d in ls)
    window_floats = frames * FEATURES + 2 * frames * CHANNELS
    params = sum(co * ci * K + co for ci, co, _ in ls)
    return dict(frames=frames, receptive=receptive, macs=macs, macs_stream=macs_stream,
                macs_cone=macs_cone, macs_window=macs_window,
                state_floats=state_floats, window_floats=window_floats, params=params)


def measured_receptive_field(frames):
    """Which input frames does the newest output depend on? Ask autograd."""
    stack = nn.ModuleList(nn.Conv1d(ci, co, K, dilation=d) for ci, co, d in layers())
    for conv in stack:
        nn.init.constant_(conv.weight, 0.01)    # all positive: no cancellation
    x = torch.zeros(1, FEATURES, frames, requires_grad=True)
    h = x
    for conv in stack:
        h = conv(F.pad(h, (conv.dilation[0] * (K - 1), 0)))   # causal: pad the left only
    h[0, :, -1].sum().backward()
    touched = (x.grad[0].abs().sum(0) > 0).nonzero().flatten()
    return len(touched), int(touched[0]), int(touched[-1])


def main():
    b = build()
    print(f"clip 1 s  -> {b['frames']} frames")
    print(f"receptive field = {b['receptive']} frames = "
          f"{ms((b['receptive'] - 1) * HOP + FRAME)} ms of audio")
    n, first, last = measured_receptive_field(b["frames"])
    print(f"autograd: the last output touches {n} input frames, {first}..{last}")
    print("MACs per layer  =", b["macs"])
    print(f"MACs per hop, streaming        = {b['macs_stream']:>9}")
    print(f"MACs per hop, recompute cone   = {b['macs_cone']:>9}")
    print(f"MACs per hop, recompute window = {b['macs_window']:>9}")
    print(f"cycles per hop at {CLOCK_HZ // 1_000_000} MHz = {CLOCK_HZ // (SAMPLE_RATE // HOP)}")
    print(f"state, streaming     = {b['state_floats']} floats = {4 * b['state_floats']} bytes")
    print(f"window, recompute    = {b['window_floats']} floats = {4 * b['window_floats']} bytes")
    print(f"weights              = {b['params']} params = {4 * b['params']} bytes f32")


def test_matches_burn():
    b = build()
    assert (b["frames"], b["receptive"]) == (98, 31)
    assert (b["macs_stream"], b["macs_cone"], b["macs_window"]) == (44_544, 751_104, 4_365_312)
    assert (b["state_floats"], b["window_floats"], b["params"]) == (1_872, 16_464, 44_800)
    assert measured_receptive_field(b["frames"]) == (31, 67, 97)


if __name__ == "__main__":
    main()
