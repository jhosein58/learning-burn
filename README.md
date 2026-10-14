# Learning Burn

<p align="center">
  <img src="img/cover.png" alt="Learning Burn — book cover" width="360">
</p>

[![Burn](https://img.shields.io/badge/burn-0.22.0-3b5b92)](https://github.com/tracel-ai/burn)
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
| "It fails at 3 a.m. with a shape error" | Tensor rank is part of the type. `Tensor<2>` and `Tensor<3>` are different types; mixing them doesn't compile. |
| "Two teams: researchers in Python, engineers rewriting in C++" | One language for both. The training code and the deployed code are the same code. |
| "Different code for CPU, GPU, and the device" | The device is a *value*. `run(&Device::flex())`, `run(&Device::wgpu(..))`, or a `no_std` build for the chip — same function body; Cargo features decide which backends are compiled in. |

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

## What's in the book

Twenty-six chapters in six parts, plus an appendix. Every idea is built by hand
first, then done the Burn way, and checked against PyTorch.

**Appendix A (`-c 0`)** — Rust for Python people: ownership, borrowing, traits,
generics. No Burn yet; read it first if Rust is new to you.

**Part I, Foundations (Chapters 1–4)** — tensors and devices · elementwise ops and
broadcasting · matmul and the shape rule, norms, determinant, Gram matrix · autodiff,
and linear regression by gradient descent.

**Part II, Learning by Hand (5–11)** — activations and *why ReLU beat sigmoid* (it's one
number: 0.25) · losses · optimizers and schedules · **backpropagation** with numbers
small enough to check on paper · a network built from scratch · attention and a
transformer block · **convolutions** from loops, with the backward pass by hand.

**Part III, Building with Burn (12–14)** — feeding the model (`Dataset`, `Batcher`,
`DataLoader`, worker threads) · modern architectures with `burn::nn` (BatchNorm,
residual blocks, a small ResNet, a transformer encoder) · the `Learner`, metrics and
checkpoints that resume exactly.

**Part IV, Generative Models (15–18)** — language models from scratch (byte-pair
tokeniser, bigram, a tiny GPT with hand-written Adam, sampling, a KV cache) and the Burn
way (a GPT with `burn::nn` and the `Learner`, trained on Shakespeare) · diffusion and flow
matching from scratch on 2-D data (DDPM, DDIM, Euler and Heun) and the Burn way (a small
U-Net with classifier-free guidance and reflow on images).

**Part V, Shipping (19–22)** — saving and importing (burnpack, `.pt` and safetensors
via `burn-store`, ONNX via `burn-onnx`) · serving: an axum server, batching and
micro-batching · bare-metal: weights in the binary, int8, the memory budget ·
**firmware in Rust**: a 28.9M-parameter language model on an ESP32-S3, from
[esp32-tinyLLM](https://github.com/Bellman281/esp32-tinyLLM) (needs an
ESP32-S3-DevKitC N16R8 for the on-chip part; everything else runs on a laptop).

**Part VI, On the Edge (23–26)** — streaming inference: causal convolutions with state
buffers, Google's `kws_streaming` modes ported to Burn, a GRU one frame at a time.
Keyword spotting, real-time denoising and Wi-Fi sensing are *in preparation*.

Full list in [`examples/`](examples/).

---

## Credit

Built on [**Burn**](https://github.com/tracel-ai/burn) (pinned to 0.22.0). The
framework is theirs; this repo is original example code against its public API.
**If this is useful to you, star [the Burn team](https://github.com/tracel-ai/burn),
not us.** They did the hard part.

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Copyright © 2026 

**Omid Safarzadeh** ([@Bellman281](https://github.com/Bellman281)),
**Hosein Jalili** ([@jhosein58](https://github.com/jhosein58))
and **Pooya Mohammadi** ([@pooya-mohammadi](https://github.com/pooya-mohammadi)).

If you adapt
these examples, please attribute <https://github.com/jhosein58/learning-burn>
— see [`NOTICE`](NOTICE).
