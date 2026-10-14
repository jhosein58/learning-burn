"""Example 23.6 - What one frame costs (PyTorch reference + test).

The untrained detector of Example 23.5 (same LCG weights), priced per frame by
formula and timed in PyTorch on this CPU: whole clip, recompute the window,
and one frame per call with the state carried outside. Its logits on the same
noise clip are the reference for rust/src/main.rs.

Run:   python main.py
Test:  pytest main.py
"""

import importlib.util
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "c23e5", os.path.join(HERE, "..", "..", "e05_streaming_detector", "python", "main.py"))
e5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e5)          # the Detector and the LCG of Example 23.5

T, K, C, H, DILATIONS = e5.T, e5.K, e5.C, e5.H, e5.DILATIONS
CLIPS = 20


def budget():
    conv_macs = sum(C * (1 if i == 0 else C) * K for i in range(len(DILATIONS)))
    conv_state = sum((1 if i == 0 else C) * d * (K - 1) for i, d in enumerate(DILATIONS))
    gru_macs, head_macs = 3 * H * (C + H), H
    params = sum(C * (1 if i == 0 else C) * K + C for i in range(len(DILATIONS)))
    params += 3 * (C * H + H + H * H + H) + H + 1
    return dict(conv_macs=conv_macs, gru_macs=gru_macs, head_macs=head_macs,
                params=params, conv_state=conv_state, gru_state=H)


@torch.no_grad()
def build(timing=False):
    model = e5.Detector()
    rng = e5.Lcg(99)
    x = torch.tensor(rng.grid(T, 32, 64.0)).reshape(1, 1, T)
    whole, streamed = e5.whole_and_streamed(model, x)
    times = None
    if timing:
        clips = [torch.tensor(rng.grid(T, 32, 64.0)).reshape(1, 1, T) for _ in range(CLIPS)]
        model(clips[0])                                        # warm-up
        start = time.perf_counter()
        for c in clips:
            model(c)
        t_whole = (time.perf_counter() - start) / (CLIPS * T)
        start = time.perf_counter()
        for t in range(T):
            window = torch.zeros(1, 1, T)
            window[:, :, T - 1 - t:] = clips[0][:, :, :t + 1]
            model(window)
        t_recompute = (time.perf_counter() - start) / T
        start = time.perf_counter()
        for c in clips:
            state = model.zero_state(1)
            for t in range(T):
                _, state = model.step(c[:, :, t:t + 1], state)
        t_stream = (time.perf_counter() - start) / (CLIPS * T)
        times = (t_whole, t_recompute, t_stream)
    return model, whole, streamed, times


def main():
    torch.set_num_threads(2)
    b = budget()
    macs = b["conv_macs"] + b["gru_macs"] + b["head_macs"]
    model, whole, streamed, times = build(timing=True)
    print(f"per frame: {macs} MACs (convs {b['conv_macs']}, GRU {b['gru_macs']}, head {b['head_macs']})")
    print(f"parameters: formula {b['params']}, counted {sum(p.numel() for p in model.parameters())}")
    print(f"state: {b['conv_state'] + b['gru_state']} floats")
    print("first five logits =", [float(f"{v:.6g}") for v in whole[0, :5].tolist()])
    print(f"max|whole - stream| = {(whole - streamed).abs().max().item():.1e}")
    print("time per frame on this CPU (PyTorch, batch 1):")
    for name, t in zip(["whole clip, amortised", "recompute the window", "stream, nn modules"], times):
        print(f"  {name:<22} {t * 1e6:>9.1f} us")


def test_matches_burn():
    b = budget()
    assert (b["conv_macs"], b["gru_macs"], b["head_macs"]) == (408, 1152, 16)
    assert (b["params"], b["conv_state"], b["gru_state"]) == (1697, 98, 16)
    model, whole, streamed, _ = build()
    assert sum(p.numel() for p in model.parameters()) == 1697
    # rust/src/main.rs computes the same logits with Burn and with plain slices.
    expected = torch.tensor([-0.000382167229, -0.000360569509, -0.000787793775, -0.00118244661, -0.00146851013])
    assert (whole[0, :5] - expected).abs().max() < 1e-8
    assert (whole - streamed).abs().max() < 1e-6


if __name__ == "__main__":
    main()
