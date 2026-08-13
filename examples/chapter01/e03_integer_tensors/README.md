# Integer Tensors

Not every tensor stores floating-point values. Burn supports integer tensors by specifying the `Int` kind.

This example shows how to build an integer tensor from existing integer data using `TensorData`.

> Use integer tensors for labels, indices, and counts—not floating-point values.
