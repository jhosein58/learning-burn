import torch
import torch.nn.functional as F

# Teacher and student scores for three samples across four classes.
student_logits = torch.tensor(
    [
        [2.0, 0.5, -1.0, 0.0],
        [0.1, 1.2, 0.3, -0.5],
        [-0.2, 0.4, 1.5, 0.8],
    ],
)

teacher_logits = torch.tensor(
    [
        [2.5, 0.2, -0.8, 0.1],
        [-0.1, 1.6, 0.2, -0.3],
        [0.0, 0.7, 1.2, 0.4],
    ],
)

# KLDiv expects log-probabilities for input and probabilities for target.
student = F.log_softmax(student_logits, dim=1)     # log-probs
teacher = F.softmax(teacher_logits, dim=1)         # probs

loss = F.kl_div(student, teacher, reduction="batchmean")

print(f"loss = {loss.item()}")  # ~0.034
