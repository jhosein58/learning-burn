"""Example 23.3 - A Stream wrapper for Conv1d, after Google's kws_streaming.

The PyTorch twin of rust/src/main.rs. One Conv1d, four modes: TRAINING and
NON_STREAM_INFERENCE run the whole clip with B zeros padded on the left;
STREAM_INTERNAL_STATE keeps the last B input frames in a registered buffer;
STREAM_EXTERNAL_STATE takes the state in and hands it back. B = max(0,
d(k-1) + 1 - s). Same LCG weights and data as the Rust side.

Run:   python main.py
Test:  pytest main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

T, C_IN, C_OUT = 24, 2, 3
CASES = [(3, 1, 1), (3, 4, 1), (5, 1, 2), (3, 3, 2), (4, 2, 3)]   # (k, d, s)
MODES = ("TRAINING", "NON_STREAM_INFERENCE",
         "STREAM_INTERNAL_STATE_INFERENCE", "STREAM_EXTERNAL_STATE_INFERENCE")


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.s >> 16) & 0x7FFF

    def grid(self, n, m, div):
        return [((self.next() % (2 * m + 1)) - m) / div for _ in range(n)]


def buffer_len(k, d, s):
    return max(0, d * (k - 1) + 1 - s)


class Stream(nn.Module):
    def __init__(self, conv, mode):
        super().__init__()
        assert mode in MODES
        self.conv, self.mode = conv, mode
        k, d, s = conv.kernel_size[0], conv.dilation[0], conv.stride[0]
        self.buffer = buffer_len(k, d, s)
        self.register_buffer("state", None)          # internal state, created lazily

    def zero_state(self, batch):
        return torch.zeros(batch, self.conv.in_channels, self.buffer)

    def forward(self, x):                            # TRAINING / NON_STREAM_INFERENCE
        assert self.mode in MODES[:2]
        return self.conv(F.pad(x, (self.buffer, 0)))

    def step(self, chunk, state):                    # STREAM_EXTERNAL_STATE_INFERENCE
        assert self.mode == MODES[3]
        return self._step(chunk, state)

    def step_internal(self, chunk):                  # STREAM_INTERNAL_STATE_INFERENCE
        assert self.mode == MODES[2]
        if self.state is None:
            self.state = self.zero_state(chunk.shape[0])
        y, self.state = self._step(chunk, self.state)
        return y

    def _step(self, chunk, state):
        window = torch.cat([state, chunk], dim=2)
        return self.conv(window), window[:, :, window.shape[2] - self.buffer:]


def conv(c_in, c_out, k, d, s, rng):
    layer = nn.Conv1d(c_in, c_out, k, stride=s, dilation=d)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(rng.grid(c_out * c_in * k, 16, 32.0)).reshape(c_out, c_in, k))
        layer.bias.copy_(torch.tensor(rng.grid(c_out, 16, 32.0)))
    return layer


@torch.no_grad()
def build():
    rng = Lcg(2020)
    x = torch.tensor(rng.grid(C_IN * T, 8, 8.0)).reshape(1, C_IN, T)
    rows, whole_first = [], None
    for case, (k, d, s) in enumerate(CASES):
        layer = conv(C_IN, C_OUT, k, d, s, rng)
        whole = Stream(layer, "NON_STREAM_INFERENCE")(x)
        internal = Stream(layer, "STREAM_INTERNAL_STATE_INFERENCE")
        y_int = torch.cat([internal.step_internal(x[:, :, j * s:(j + 1) * s])
                           for j in range(T // s)], dim=2)
        external = Stream(layer, "STREAM_EXTERNAL_STATE_INFERENCE")
        state, ys = external.zero_state(1), []
        for j in range(T // s):
            y, state = external.step(x[:, :, j * s:(j + 1) * s], state)
            ys.append(y)
        y_ext = torch.cat(ys, dim=2)
        rows.append((k, d, s, external.buffer, list(whole.shape),
                     (whole - y_int).abs().max().item(), (whole - y_ext).abs().max().item()))
        if case == 0:
            whole_first = whole[0, 0, :6].tolist()

    stack = [Stream(conv(ci, co, 3, d, 1, rng), "TRAINING")
             for ci, co, d in [(C_IN, 4, 1), (4, 4, 2), (4, 4, 4)]]
    whole = x
    for i, layer in enumerate(stack):
        whole = layer(whole)
        whole = F.relu(whole) if i + 1 < len(stack) else whole
    stack = [Stream(l.conv, "STREAM_EXTERNAL_STATE_INFERENCE") for l in stack]
    states, ys = [l.zero_state(1) for l in stack], []
    for t in range(T):
        h = x[:, :, t:t + 1]
        for i, layer in enumerate(stack):
            y, states[i] = layer.step(h, states[i])
            h = F.relu(y) if i + 1 < len(stack) else y
        ys.append(h)
    streamed = torch.cat(ys, dim=2)
    tcn_diff = (whole - streamed).abs().max().item()
    return rows, whole_first, tcn_diff, whole[0, 0, :6].tolist(), sum(s.numel() for s in states)


def main():
    rows, whole_first, tcn_diff, tcn_first, floats = build()
    print(" k  d  s  B  shape        max|diff| int  ext")
    for k, d, s, b, shape, di, de in rows:
        print(f"{k:>2} {d:>2} {s:>2} {b:>2}  {str(shape):<12} {di:>12g} {de:>4g}")
    print(f"TCN: external state = {floats} floats, max|whole - stream| = {tcn_diff:g}")
    print("case 0 y[0][:6] =", [round(v, 8) for v in whole_first])
    print("TCN    y[0][:6] =", [round(v, 8) for v in tcn_first])


def test_matches_burn():
    rows, whole_first, tcn_diff, tcn_first, floats = build()
    assert [r[3] for r in rows] == [2, 8, 3, 5, 4]
    assert all(di == 0 and de == 0 for *_, di, de in rows)
    assert tcn_diff == 0 and floats == 52
    # rust/src/main.rs prints the same numbers.
    expected_tcn = [-0.50109863, -0.48828125, -0.5571594, -0.4544983, -0.70116425, -0.3212204]
    assert max(abs(a - b) for a, b in zip(tcn_first, expected_tcn)) < 1e-6
    assert whole_first == [0.171875, 0.6484375, -0.4296875, -0.1640625, 0.45703125, 0.66015625]


if __name__ == "__main__":
    main()
