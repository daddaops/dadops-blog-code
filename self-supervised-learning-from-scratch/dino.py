"""DINO (Distillation with NO labels) — self-supervised ViT training."""
import numpy as np

def softmax(x, temp=1.0):
    """Temperature-scaled softmax."""
    e = np.exp((x - np.max(x, axis=-1, keepdims=True)) / temp)
    return e / e.sum(axis=-1, keepdims=True)

def dino_loss(student_out, teacher_out, center, temp_s=0.1, temp_t=0.04):
    """DINO loss: cross-entropy between sharpened teacher and student."""
    # Teacher: sharpen via low temperature + subtract center
    t_probs = softmax(teacher_out - center, temp=temp_t)
    # Student: higher temperature (softer distribution)
    s_log_probs = np.log(softmax(student_out, temp=temp_s) + 1e-10)
    # Cross-entropy: teacher distribution is the "label"
    return -np.mean(np.sum(t_probs * s_log_probs, axis=-1))

# Setup: 4 images, each gets multiple crops at different scales
np.random.seed(42)
dim = 16
batch_size = 4
n_global = 2    # teacher processes global crops (large image regions)
n_local = 4     # student processes all crops including small local ones

# Simulate encoder outputs (normally these come from a ViT)
student_global = np.random.randn(batch_size, n_global, dim) * 0.5
student_local = np.random.randn(batch_size, n_local, dim) * 0.5
teacher_global = student_global + np.random.randn(batch_size, n_global, dim) * 0.02

# Center: running mean of teacher outputs (prevents mode collapse)
center = np.zeros(dim)
center_ema = 0.9

# DINO objective: every student crop predicts every teacher global crop
total_loss = 0.0
n_pairs = 0

for img in range(batch_size):
    for t in range(n_global):
        teacher_out = teacher_global[img, t]
        # Student global crops (skip matching same crop index)
        for s in range(n_global):
            if s == t:
                continue
            total_loss += dino_loss(student_global[img, s], teacher_out, center)
            n_pairs += 1
        # Student local crops (always included)
        for s in range(n_local):
            total_loss += dino_loss(student_local[img, s], teacher_out, center)
            n_pairs += 1

# Update center with EMA
batch_center = teacher_global.reshape(-1, dim).mean(axis=0)
center = center_ema * center + (1 - center_ema) * batch_center

avg_loss = total_loss / n_pairs
t_sharp = softmax(teacher_global[0, 0], temp=0.04)
t_entropy = -np.sum(t_sharp * np.log(t_sharp + 1e-10))

print(f"DINO loss: {avg_loss:.4f}")
print(f"Loss pairs per image: {n_pairs // batch_size}")
print(f"  (= {n_global}x{n_global-1} global-global + {n_global}x{n_local} global-local)")
print(f"Teacher sharpness (entropy): {t_entropy:.3f} (lower = sharper)")
print(f"Center norm: {np.linalg.norm(center):.4f}")
