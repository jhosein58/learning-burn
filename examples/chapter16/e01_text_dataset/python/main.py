"""Example 16.1 - A text dataset (PyTorch reference + parity test).

The same character vocabulary, split and fixed-length windows as rust/src/main.rs.
The corpus is a 200 KB slice of Tiny Shakespeare (public domain, via Karpathy's char-rnn).

Run:   python main.py
Test:  pytest main.py
"""

import math
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                    "shakespeare.txt")
BLOCK = 128   # context length T
BATCH = 16    # batch size N


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))                 # id -> character
        self.ids = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.ids[c] for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


class TextWindows(Dataset):
    """Window i is tokens[i*stride : i*stride + T + 1]."""

    def __init__(self, tokens, block, stride):
        self.tokens, self.block, self.stride = tokens, block, stride

    def __len__(self):
        return (len(self.tokens) - self.block - 1) // self.stride + 1

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        start = i * self.stride
        w = torch.tensor(self.tokens[start:start + self.block + 1])
        return w[:-1], w[1:]        # input, target (shifted by one)


def split(tokens):
    cut = len(tokens) * 9 // 10
    return tokens[:cut], tokens[cut:]


def entropies(tokens, vocab):
    """Unigram and bigram entropy of the next character, in nats, over the given text."""
    one, two = Counter(tokens), Counter(zip(tokens, tokens[1:]))
    n = len(tokens)
    h1 = -sum(c / n * math.log(c / n) for c in one.values())
    h2 = -sum(c / (n - 1) * math.log(c / one[a]) for (a, _), c in two.items())
    return h1, h2


def build():
    text = open(DATA).read()
    tok = CharTokenizer(text)
    tokens = tok.encode(text)
    train_ids, valid_ids = split(tokens)
    train = TextWindows(train_ids, BLOCK, BLOCK)
    valid = TextWindows(valid_ids, BLOCK, BLOCK)

    inputs, targets = next(iter(DataLoader(train, batch_size=BATCH, shuffle=False)))
    shifted = torch.equal(inputs[:, 1:], targets[:, :-1])

    loader = DataLoader(range(len(train)), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(16))
    epoch1 = [int(i) * BLOCK for b in loader for i in b]
    epoch2 = [int(i) * BLOCK for b in loader for i in b]
    all_once = sorted(epoch1) == [i * BLOCK for i in range(len(train))]
    return dict(chars=len(text), vocab="".join(tok.chars), encoded=tok.encode("First Citizen:"),
                round_trip=tok.decode(tokens) == text, split=(len(train_ids), len(valid_ids)),
                entropy=entropies(train_ids, len(tok.chars)),
                windows=(len(train), len(valid)),
                window0=(tok.decode(train[0][0].tolist()), tok.decode(train[0][1].tolist())),
                batch_dims=(list(inputs.shape), list(targets.shape)), shifted=shifted,
                batch_sum=int(inputs.sum()), epochs=(epoch1, epoch2, all_once))


def main():
    s = build()
    print(f"text       = {s['chars']} characters")
    print(f"vocabulary = {len(s['vocab'])} symbols: {s['vocab']!r}")
    print(f"encode(\"First Citizen:\") = {s['encoded']}")
    print(f"decode(encode(text)) == text: {s['round_trip']}")
    print(f"split      = {s['split'][0]} train / {s['split'][1]} valid characters")
    print(f"entropy    = {s['entropy'][0]:.4f} nats per character alone, "
          f"{s['entropy'][1]:.4f} given the one before")
    print(f"windows    = {s['windows'][0]} train / {s['windows'][1]} valid  "
          f"(T = {BLOCK}, stride {BLOCK})")
    print(f"window 0 input  = {s['window0'][0][:32]!r}...")
    print(f"window 0 target = {s['window0'][1][:32]!r}...")
    print(f"first batch = inputs {s['batch_dims'][0]}, targets {s['batch_dims'][1]}, "
          f"sum of input ids {s['batch_sum']}")
    print(f"targets[:, t] == inputs[:, t + 1]: {s['shifted']}")
    print(f"epoch 1 starts with windows at {s['epochs'][0][:4]}")
    print(f"epoch 2 starts with windows at {s['epochs'][1][:4]}")
    print(f"each epoch sees every training window exactly once: {s['epochs'][2]}")


def test_matches_burn():
    s = build()
    # From rust/src/main.rs. The shuffled orders differ (different RNGs); the guarantees do not.
    assert s["chars"] == 199994
    assert len(s["vocab"]) == 62
    assert s["encoded"] == EXPECTED_ENCODED
    assert s["round_trip"]
    assert s["split"] == (179994, 20000)
    assert abs(s["entropy"][0] - 3.297924) < 1e-6 and abs(s["entropy"][1] - 2.398677) < 1e-6
    assert s["windows"] == (1406, 156)
    assert s["batch_dims"] == ([16, 128], [16, 128])
    assert s["shifted"]
    assert s["batch_sum"] == EXPECTED_BATCH_SUM
    assert s["epochs"][2] and s["epochs"][0] != s["epochs"][1]


EXPECTED_ENCODED = [16, 44, 53, 54, 55, 1, 13, 44, 55, 44, 61, 40, 49, 8]
EXPECTED_BATCH_SUM = 73336


if __name__ == "__main__":
    main()
