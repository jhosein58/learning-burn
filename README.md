# Learning Burn

[![Burn](https://img.shields.io/badge/burn-0.21.0-3b5b92)](https://github.com/tracel-ai/burn)
[![Rust](https://img.shields.io/badge/rust-2024_edition-dea584?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-omidpolyO-FFDD00?logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/omidpolyO)

Deep learning in Rust, for people who already do it in Python.

Runnable Rust and Python examples for every idea in the *Learning Burn* book. Use the built-in CLI to launch any example by chapter, example number, and language.

---

## Why would a data scientist touch this?

Most of you shouldn't, most of the time. If your job is EDA, feature work, and
iterating on models in a notebook, **stay in Python.** The tooling is better, the
ecosystem is thirty times larger, and nothing here will make you faster at that.

Rust becomes the right answer at exactly one moment: **when the model has to leave
your laptop.**

| Your problem | What Rust gives you |
|---|---|
| "Inference is fast, but the *service* is slow" | No interpreter, no GIL, no GC pauses. Predictable p99 latency. |
| "Deployment is a 4 GB Docker image and a dependency hell" | One static binary. No Python, no CUDA runtime, no `requirements.txt`. |
| "It needs to run on the device, not the cloud" | `no_std`. Weights compiled into the firmware. **A 2,410-parameter model is 2.4 KB quantised** — it fits on a microcontroller with 256 KB of SRAM. |
| "It fails at 3 a.m. with a shape error" | Tensor rank is part of the type. `Tensor<B, 2>` and `Tensor<B, 3>` are different types; mixing them doesn't compile. |
| "Two teams: researchers in Python, engineers rewriting in C++" | One language for both. The training code and the deployed code are the same code. |
| "Different code for CPU, GPU, and the device" | The backend is a *type parameter*. `run::<NdArray>()`, `run::<Wgpu>()`, `run::<Chip>()` — same function body, chosen at the call site. |

If none of those are your problem, close the tab — with our blessing. If one of
them is the reason your last project stalled, this repo is worth an afternoon.

**What Rust will not give you:** a mature ecosystem, pretrained model zoos,
`pandas`, or a fast iteration loop. Burn is good. It is not PyTorch, and pretending
otherwise would waste your time.

---

## Running examples

Clone the repository and navigate to the `learning-burn` directory.

```bash
git clone https://github.com/jhosein58/learning-burn
cd learning-burn
```

Examples are organized by chapter and example number.

```bash
cargo run -- -c <chapter> -e <example> <language>
```


Example:

```bash
cargo run -- -c 2 -e 4 rust
cargo run -- -c 3 -e 1 python
```

---

## What's in the sixteen chapters

**Foundations** — Rust for Python people (ownership, borrowing, traits — no Burn
yet) · tensors · elementwise ops and broadcasting · matmul and the shape rule ·
norms, determinant, Gram matrix · autodiff.

**Learning** — linear regression by hand · activations and *why ReLU beat sigmoid*
(it's one number: 0.25) · MSE and cross-entropy · SGD and the learning rate ·
**backpropagation** — the chain rule with numbers small enough to check on paper ·
a network built from scratch, with the gradients derived by hand and then verified
against autodiff.

**Building** — attention, causal masking, multi-head, a full transformer block.

**Shipping** — save/load and the f32-vs-f16 trade-off · bare-metal: weights baked
into the binary, backend-agnostic inference, int8 quantisation, memory footprint ·
running pretrained models with Candle, and when to use it instead of Burn.

Full list in [`examples/`](examples/).

---

## Credit

Built on [**Burn**](https://github.com/tracel-ai/burn) (pinned to 0.21.0). The
framework is theirs; this repo is original example code against its public API.
**If this is useful to you, star [the Burn team](https://github.com/tracel-ai/burn),
not us.** They did the hard part.

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Copyright © 2026 **Bellman281** and
**H0531N**. If you adapt these examples, please attribute
<https://github.com/jhosein58/learning-burn> — see [`NOTICE`](NOTICE).
