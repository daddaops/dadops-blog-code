import numpy as np
from training import teacher, student_kd, X_train, softmax_batch, MLP

class FitNetRegressor:
    """
    Projects student features into teacher feature space.
    student_dim → teacher_dim via a learned linear map.
    """
    def __init__(self, student_dim, teacher_dim):
        scale = np.sqrt(2.0 / student_dim)
        self.W = np.random.randn(student_dim, teacher_dim) * scale
        self.b = np.zeros(teacher_dim)

    def forward(self, student_features):
        """Project student features into teacher space."""
        return student_features @ self.W + self.b

    def backward(self, student_features, d_projected, lr=0.01):
        """Update regressor weights and return gradient for student."""
        batch = student_features.shape[0]
        dW = student_features.T @ d_projected / batch
        db = np.mean(d_projected, axis=0)
        self.W -= lr * dW
        self.b -= lr * db
        # Gradient flowing back to the student's hidden layer
        return d_projected @ self.W.T

def hint_loss(teacher_features, student_features, regressor):
    """
    FitNets hint loss: MSE between teacher features and
    projected student features.

    teacher_features: hidden activations from teacher — (batch, teacher_dim)
    student_features: hidden activations from student — (batch, student_dim)
    regressor:        FitNetRegressor mapping student → teacher space
    """
    projected = regressor.forward(student_features)  # (batch, teacher_dim)
    diff = projected - teacher_features               # (batch, teacher_dim)
    loss = np.mean(diff ** 2)                         # scalar

    # Gradient of MSE w.r.t. projected features
    d_projected = 2.0 * diff / (diff.shape[0] * diff.shape[1])

    return loss, d_projected

# Example: match features from our teacher and student
teacher.forward(X_train[:5])
teacher_feats = teacher.h            # (5, 128)

student_kd.forward(X_train[:5])
student_feats = student_kd.h         # (5, 32)

reg = FitNetRegressor(32, 128)
hloss, d_proj = hint_loss(teacher_feats, student_feats, reg)
print(f"Hint loss: {hloss:.4f}")
# Hint loss: 2.3451  (high initially — student and teacher features don't match yet)
