"""Example 15.2 - A Bigram Model by Counting (PyTorch reference + parity test).

Count how often each byte follows each other byte in the training text,
normalise every row, and the table is a language model. Its quality is the
mean negative log-likelihood (NLL) of held-out text, or its exponential,
the perplexity.

Run:   python main.py
Test:  pytest main.py
"""

import math
from pathlib import Path

import torch

TEXT = (Path(__file__).resolve().parents[2] / "data" / "shakespeare.txt").read_bytes()


def count_pairs(ids, v):
    c = torch.zeros(v, v)
    a, b = torch.tensor(ids[:-1]), torch.tensor(ids[1:])
    c.index_put_((a, b), torch.ones(len(ids) - 1), accumulate=True)
    return c


def probabilities(counts, alpha):
    v = counts.shape[1]
    return (counts + alpha) / (counts.sum(dim=1, keepdim=True) + alpha * v)


def nll(eval_counts, probs):
    logp = probs.log().masked_fill(eval_counts == 0, 0.0)
    return (-(eval_counts * logp).sum() / eval_counts.sum()).item()


def show(b):
    return {32: "_", 10: "\\n"}.get(b, chr(b))


def build(verbose=False):
    voc = sorted(set(TEXT))
    v = len(voc)
    index = {b: i for i, b in enumerate(voc)}
    ids = [index[b] for b in TEXT]
    n_train = len(ids) * 9 // 10
    train = count_pairs(ids[:n_train], v)
    val = count_pairs(ids[n_train:], v)
    never = train == 0
    unseen = (int((never & (val > 0)).sum()), int(val[never].sum()))
    mle = probabilities(train, 0.0)
    if verbose:
        print(f"vocab {v}, train pairs {int(train.sum())}, val pairs {int(val.sum())}")
        for ctx in b"qT ":
            p, idx = mle[index[ctx]].sort(descending=True)
            top = [f"{show(voc[idx[j]])} {p[j]:.3f}" for j in range(3) if p[j] > 0]
            print(f"after {show(ctx):<2} -> " + ", ".join(top))
    floor = nll(train, mle)
    table = []
    for alpha in [1.0, 0.1, 0.01]:
        p = probabilities(train, alpha)
        table.append((alpha, nll(train, p), nll(val, p)))
    return floor, table, unseen


def main():
    floor, table, (kinds, times) = build(verbose=True)
    print(f"pairs seen in val but never in train: {kinds} kinds, {times} times")
    print(f"alpha 0 (pure counts): train NLL {floor:.4f}, perplexity {math.exp(floor):.2f}; val NLL inf")
    print("alpha  train NLL  val NLL  val perplexity")
    for alpha, tr, va in table:
        print(f"{alpha:<5g}  {tr:.4f}     {va:.4f}   {math.exp(va):.2f}")
    print(f"uniform: NLL ln 62 = {math.log(62):.4f}, perplexity 62")


def test_matches_burn():
    floor, table, unseen = build()
    assert unseen == (17, 26)
    assert abs(floor - 2.3987) < 1e-4
    burn = [(2.4122, 2.4915), (2.4002, 2.4820), (2.3988, 2.4833)]
    assert all(abs(tr - a) < 1e-4 and abs(va - b) < 1e-4
               for (_, tr, va), (a, b) in zip(table, burn))


if __name__ == "__main__":
    main()
