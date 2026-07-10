# Learning Burn — Code Examples

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-omidpolyO-FFDD00?logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/omidpolyO)

Compilable, tested code examples that accompany the **Learning Burn** book
(by H0531N and Bellman281).

Every example is a small, self-contained program you can build and run while
reading. The examples are organized by chapter and are each verified with
`cargo build --examples` and `cargo run`, so what you see in the book is what
actually compiles.

## Built on Burn

These examples are written for [**Burn**](https://github.com/tracel-ai/burn), the
Rust deep learning framework. All credit for the framework itself goes to the
Burn authors and contributors. This repository contains **original example
code** demonstrating Burn's public API — it does not copy Burn's source.

## Running the examples

```bash
cd book-tests
cargo build --examples                     # compile every example
cargo run --example ch1_02_basic_creation  # run a single example
```

## Examples by chapter

| Chapter | Topic | Examples |
|---|---|---|
| 0 | Rust for Python People | variables, ownership, borrowing, iterators, structs, enums/Option, Result, traits, generics |
| 1 | Creating Tensors | `ch1_01` … `ch1_09` |
| 2 | Tensor Operations | arithmetic, broadcasting, reshaping/slicing, reductions, standardize, masking |
| 3 | Matrix Algebra | matmul, transpose/matvec/outer, structured & batched, linear layer |
| 4 | Linear Algebra | vector norms, Gram matrix |
| 5 | Autodiff | backward gradient, gradient container, `inner()` for inference |
| 6 | Linear Regression | manual gradient descent, autodiff gradient |
| 7 | Activation Functions | ReLU family, saturating, softmax |
| 8 | Loss Functions | MSE, cross-entropy |
| 9 | Optimizers | SGD training loop |
| 10 | Backpropagation | chain rule, two-layer backprop, XOR MLP |
| 11 | Neural Network from Scratch | forward from scratch, manual vs autodiff backward, from-scratch training, gradient check |
| 12 | Attention & Transformers | scaled dot-product attention, causal masking, multi-head attention, transformer block |

## License & attribution

Licensed under the Apache License, Version 2.0 — see [`LICENSE`](LICENSE).

Copyright © 2026 **H0531N** and **Bellman281**.

These examples accompany the *Learning Burn* book. If you use or adapt them,
please attribute this repository:
<https://github.com/jhosein58/learning-burn> — see [`NOTICE`](NOTICE).
