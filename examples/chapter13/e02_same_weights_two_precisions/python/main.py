import torch
import torch.nn as nn
import os
import tempfile

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


x = torch.tensor([[2.0]])
model = Model()
original = model(x)

# --- full precision (f32) — exact round trip ---
full_path = os.path.join(tempfile.gettempdir(), "ch13_full.pt")
torch.save(model.state_dict(), full_path)
from_full = Model()
from_full.load_state_dict(torch.load(full_path, weights_only=True))
pred_full = from_full(x)  # exactly equal

# --- half precision (f16) — smaller, slightly lossy ---
half_path = os.path.join(tempfile.gettempdir(), "ch13_compact.pt")
torch.save(model.half().state_dict(), half_path)
from_compact = Model().half()
from_compact.load_state_dict(torch.load(half_path, weights_only=True))
pred_compact = from_compact(x.half())  # very close

full_size = os.path.getsize(full_path)
compact_size = os.path.getsize(half_path)

print(f"original prediction   = [{original.item()}]")
print(f"full-precision reload = [{pred_full.item()}]")     # exactly equal
print(f"compact (f16) reload  = [{pred_compact.item()}]")  # very close
print(f"full    (f32) file    = {full_size} bytes")
print(f"compact (f16) file    = {compact_size} bytes")
