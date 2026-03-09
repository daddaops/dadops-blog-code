import numpy as np


class SimCLREncoder:
    """Encoder + projection head (reused for DINO student/teacher)."""

    def __init__(self, input_dim, hidden_dim=64, proj_dim=32):
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        scale = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, proj_dim) * scale
        self.b2 = np.zeros(proj_dim)

    def encode(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h

    def project(self, h):
        z = h @ self.W2 + self.b2
        z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        return z

    def forward(self, x):
        h = self.encode(x)
        z = self.project(h)
        return h, z


class DINOTrainer:
    """Self-distillation with no labels — the DINO framework."""

    def __init__(self, student, teacher, output_dim,
                 center_mom=0.9, teacher_temp=0.04,
                 student_temp=0.1, ema_mom=0.996):
        self.student = student
        self.teacher = teacher
        self.center = np.zeros(output_dim)
        self.center_mom = center_mom
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.ema_mom = ema_mom

    def update_teacher(self):
        """EMA update: teacher slowly follows the student."""
        m = self.ema_mom
        for tp, sp in zip(
            [self.teacher.W1, self.teacher.b1, self.teacher.W2, self.teacher.b2],
            [self.student.W1, self.student.b1, self.student.W2, self.student.b2]
        ):
            tp[:] = m * tp + (1 - m) * sp

    def update_center(self, teacher_out):
        """Running mean prevents mode collapse."""
        self.center = (self.center_mom * self.center
                       + (1 - self.center_mom) * teacher_out.mean(axis=0))

    def compute_loss(self, student_out, teacher_out):
        """Cross-entropy between sharp teacher and soft student."""
        # Center and sharpen teacher — this is the collapse prevention
        t = (teacher_out - self.center) / self.teacher_temp
        t = np.exp(t) / np.sum(np.exp(t), axis=1, keepdims=True)

        # Student softmax (higher temperature, no centering)
        s = student_out / self.student_temp
        s_log = s - np.log(np.sum(np.exp(s), axis=1, keepdims=True))

        # Cross-entropy: H(teacher, student) = -sum(t * log(s))
        return -np.sum(t * s_log, axis=1).mean()


if __name__ == "__main__":
    np.random.seed(42)

    input_dim, hidden_dim, proj_dim = 2, 16, 8

    # Create student and teacher with same initial weights
    student = SimCLREncoder(input_dim, hidden_dim, proj_dim)
    teacher = SimCLREncoder(input_dim, hidden_dim, proj_dim)
    # Copy student weights to teacher
    teacher.W1[:] = student.W1.copy()
    teacher.b1[:] = student.b1.copy()
    teacher.W2[:] = student.W2.copy()
    teacher.b2[:] = student.b2.copy()

    trainer = DINOTrainer(student, teacher, output_dim=proj_dim)

    # Create toy data
    data = np.random.randn(32, input_dim)

    # Simulate a few training steps
    print("DINO training simulation:")
    for step in range(10):
        # Two augmented views
        view1 = data + np.random.randn(*data.shape) * 0.3
        view2 = data + np.random.randn(*data.shape) * 0.3

        # Forward pass (use raw projections, not L2-normalized, for DINO)
        s_h = student.encode(view1)
        student_out = s_h @ student.W2 + student.b2  # raw logits

        t_h = teacher.encode(view2)
        teacher_out = t_h @ teacher.W2 + teacher.b2  # raw logits

        loss = trainer.compute_loss(student_out, teacher_out)

        # Update center and teacher
        trainer.update_center(teacher_out)
        trainer.update_teacher()

        if step % 2 == 0:
            print(f"  Step {step:2d} | Loss: {loss:.4f} | Center norm: {np.linalg.norm(trainer.center):.4f}")

    print("\nDINO components verified successfully.")
