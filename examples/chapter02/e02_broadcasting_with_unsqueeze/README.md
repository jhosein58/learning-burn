# Broadcasting with unsqueeze

PyTorch vs Burn: adding a bias vector to every row of a matrix. PyTorch broadcasts
implicitly; Burn requires you to raise the vector's rank with `unsqueeze()`.

This is because rank is part of the tensor's type in Burn—a rank-1 tensor cannot
silently become rank-2. The explicit call makes your intent unmistakable.

> Being explicit is a feature, not a tax. It prevents silent shape bugs.
