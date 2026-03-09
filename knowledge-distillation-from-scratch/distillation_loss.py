import numpy as np
from soft_labels import softmax, teacher_logits

def kl_divergence(p, q):
    """KL(p || q) — how much information is lost using q to approximate p."""
    # Avoid log(0) with a small epsilon
    eps = 1e-8
    return np.sum(p * np.log((p + eps) / (q + eps)))

def cross_entropy(targets_onehot, predictions):
    """Standard cross-entropy loss."""
    eps = 1e-8
    return -np.sum(targets_onehot * np.log(predictions + eps))

def distillation_loss(teacher_logits, student_logits, true_labels, T=4.0, alpha=0.9):
    """
    Complete knowledge distillation loss.

    teacher_logits: raw logits from the (frozen) teacher  — (num_classes,)
    student_logits: raw logits from the student           — (num_classes,)
    true_labels:    one-hot ground truth                  — (num_classes,)
    T:              temperature for softening distributions
    alpha:          weight on the soft-target loss (0 to 1)
    """
    # Soft targets: both teacher and student at temperature T
    p_teacher_soft = softmax(teacher_logits, T=T)
    p_student_soft = softmax(student_logits, T=T)

    # Hard predictions: student at T=1
    p_student_hard = softmax(student_logits, T=1.0)

    # Component 1: soft-target KL divergence, scaled by T²
    soft_loss = T * T * kl_divergence(p_teacher_soft, p_student_soft)

    # Component 2: standard cross-entropy with true labels
    hard_loss = cross_entropy(true_labels, p_student_hard)

    # Weighted combination
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return total_loss

# Example: teacher and student see the same digit "2"
# Student gets the right class but has wrong structural assumptions
# (thinks 5 is similar to 2, doesn't recognize 3's or 7's similarity)
student_logits = np.array([0.5, -0.3, 3.5, 0.2, 0.8, 2.0, 0.1, -0.5, 0.3, 1.2])
true_label = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

loss = distillation_loss(teacher_logits, student_logits, true_label, T=4.0, alpha=0.9)
print(f"Distillation loss: {loss:.4f}")
# Distillation loss: 1.0229

# For comparison, standard CE (no teacher):
hard_only = cross_entropy(true_label, softmax(student_logits, T=1.0))
print(f"Hard-label CE only: {hard_only:.4f}")
# Hard-label CE only: 0.4650
