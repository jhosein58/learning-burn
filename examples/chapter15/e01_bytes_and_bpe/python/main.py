"""Example 15.1 - Bytes and BPE by Hand (Python reference + parity test).

Bytes are tokens: 256 of them, and every text is already a sequence of them.
Byte-pair encoding then merges the most frequent adjacent pair into a new
token, again and again. Same tie-break as the Rust example (the smallest
pair wins a tie), so both learn the same merges in the same order.

Run:   python main.py
Test:  pytest main.py
"""

from collections import Counter
from pathlib import Path

import torch

TEXT = (Path(__file__).resolve().parents[2] / "data" / "shakespeare.txt").read_bytes()
MERGES = 48
SENTENCE = "First Citizen:\nBefore we proceed any further, hear me speak."


def most_frequent_pair(ids):
    counts = Counter(zip(ids, ids[1:]))
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))


def merge(ids, pair, new):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text, n_merges):
    ids = list(text)
    vocab = [bytes([b]) for b in range(256)]
    merges, lengths = [], [len(ids)]
    for k in range(n_merges):
        pair, count = most_frequent_pair(ids)
        ids = merge(ids, pair, 256 + k)
        vocab.append(vocab[pair[0]] + vocab[pair[1]])
        merges.append((pair, count))
        lengths.append(len(ids))
    return merges, vocab, lengths


def encode(merges, text, k):
    ids = list(text.encode())
    for j, (pair, _) in enumerate(merges[:k]):
        ids = merge(ids, pair, 256 + j)
    return ids


def decode(vocab, ids):
    return b"".join(vocab[i] for i in ids).decode()


def show(vocab, ids):
    return "|".join(decode(vocab, [i]).replace(" ", "_").replace("\n", "\\n") for i in ids)


def main():
    print(f"bytes: {len(TEXT)} tokens, vocabulary 256 ({len(set(TEXT))} used)")
    merges, vocab, lengths = train(TEXT, MERGES)
    for k, ((a, b), count) in enumerate(merges[:12]):
        new = 256 + k
        sa, sb, sab = show(vocab, [a]), show(vocab, [b]), show(vocab, [new])
        print(f"merge {k + 1:>2}: {sa:>2} + {sb:<2} -> {new} {sab:<3} ({count} times)")
    print("merges  vocab  tokens  bytes/token")
    for k in [0, 12, 24, 48]:
        n = lengths[k]
        print(f"{k:>6}  {256 + k:>5}  {n:>6}  {len(TEXT) / n:.3f}")
    for k in [12, 48]:
        ids = encode(merges, SENTENCE, k)
        print(f"{k:>2} merges, {len(ids)} tokens: {show(vocab, ids)}")
    ids = encode(merges, SENTENCE, MERGES)
    print("round trip:", str(decode(vocab, ids) == SENTENCE).lower())
    unseen = encode(merges, "Rustacean", MERGES)
    print(f"unseen word: {show(vocab, unseen)} -> \"{decode(vocab, unseen)}\"")
    t = torch.tensor(ids)
    print(f"as a tensor: {list(t.shape)} {t[:8].tolist()}")


def test_matches_burn():
    merges, vocab, lengths = train(TEXT, MERGES)
    assert [m[0] for m in merges[:6]] == [(101, 32), (116, 104), (116, 32),
                                          (115, 32), (111, 117), (100, 32)]
    assert lengths[MERGES] == 138684
    assert decode(vocab, encode(merges, SENTENCE, MERGES)) == SENTENCE
    assert decode(vocab, encode(merges, "Rustacean", MERGES)) == "Rustacean"


if __name__ == "__main__":
    main()
