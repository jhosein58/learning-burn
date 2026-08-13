import numpy as np

w = np.array([0.5, -1.2, 0.03, 0.9, -0.4], dtype=np.float32)

scale = np.abs(w).max() / 127
q     = np.round(w / scale).astype(np.int8)  # one byte each
w2    = q * scale                            # back to float

print(np.abs(w - w2).max()) # ~0.003
