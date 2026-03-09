import numpy as np
from soft_labels import softmax, teacher_logits

def kl_gradient_wrt_student(teacher_logits, student_logits, T):
    """
    Gradient of KL(p_teacher^T || p_student^T) w.r.t. student_logits.
    Uses the analytical result: dKL/dz_j = (1/T)(p_student_j - p_teacher_j)
    """
    p_teacher = softmax(teacher_logits, T=T)
    p_student = softmax(student_logits, T=T)
    return (1.0 / T) * (p_student - p_teacher)

# Same logits as before
student_logits = np.array([0.5, -0.3, 3.5, 0.2, 0.8, 2.0, 0.1, -0.5, 0.3, 1.2])

# Measure gradient norms at different temperatures
print("T   | grad norm (KL)  | grad norm (T²·KL) | ratio to T=1")
print("----|-----------------|--------------------|--------------")
norm_at_T1 = np.linalg.norm(kl_gradient_wrt_student(teacher_logits, student_logits, T=1))
for T in [1, 2, 4, 8, 16]:
    grad = kl_gradient_wrt_student(teacher_logits, student_logits, T=T)
    raw_norm = np.linalg.norm(grad)
    scaled_norm = np.linalg.norm(T * T * grad)  # multiply by T²
    print(f"T={T:2d} | {raw_norm:.6f}      | {scaled_norm:.6f}         | "
          f"{raw_norm / norm_at_T1:.4f}")

# T   | grad norm (KL)  | grad norm (T²·KL) | ratio to T=1
# ----|-----------------|--------------------|--------------
# T= 1 | 0.298980      | 0.298980         | 1.0000
# T= 2 | 0.127134      | 0.508538         | 0.4252
# T= 4 | 0.031069      | 0.497098         | 0.1039
# T= 8 | 0.007476      | 0.478438         | 0.0250
# T=16 | 0.001837      | 0.470308         | 0.0061
