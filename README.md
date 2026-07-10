# Learning Burn — Code Examples

Compilable, tested code examples that accompany the **Learning Burn** book.

Every example is a small, self-contained program you can build and run while
reading. The examples are organized by chapter and are each verified with
`cargo build --examples` and `cargo run`, so what you see in the book is what
actually compiles.

## Built on Burn

These examples are written for [**Burn**](https://github.com/tracel-ai/burn), the
Rust deep learning framework. All credit for the framework itself goes to the
Burn authors and contributors. This repository contains **original example
code** demonstrating Burn's public API — it does not copy Burn's source. Burn is
dual-licensed under
[MIT](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT) and
[Apache-2.0](https://github.com/tracel-ai/burn/blob/main/LICENSE-APACHE), and
this repository follows the same dual-license model (see [License](#license)).

## Running the examples

```bash
cd book-tests
cargo build --examples                     # compile every example
cargo run --example ch1_02_basic_creation  # run a single example
```

## Examples by chapter

| Chapter | Topic | Examples |
|---|---|---|
| 1 | Creating Tensors | `ch1_01` … `ch1_09` |
| 2 | Tensor Operations | arithmetic, broadcasting, reshaping/slicing, reductions, standardize, masking |
| 3 | Matrix Algebra | matmul, transpose/matvec/outer, structured & batched, linear layer |
| 4 | Linear Algebra | vector norms, Gram matrix |
| 5 | Autodiff | backward gradient, gradient container, `inner()` for inference |
| 6 | Linear Regression | manual gradient descent, autodiff gradient |
| 7 | Activation Functions | ReLU family, saturating, softmax |
| 8 | Loss Functions | MSE, cross-entropy |
| 9 | Optimizers | SGD training loop |

## License

This repository is dual-licensed, matching the Burn framework, under either of:

- MIT License ([`LICENSE`](LICENSE))
- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))

at your option. © 2026 H0531N.
