# Matrix Multiplication (matmul)

PyTorch vs Burn: the matrix product. PyTorch uses `@`; Burn uses `.matmul()`.

The inner dimensions must agree — they are summed over and vanish, leaving
the two outer dimensions behind.

> Every cell is the dot product of a row from A and a column from B.
> This is not element-wise `*` — that pairing was covered in Chapter 2.
