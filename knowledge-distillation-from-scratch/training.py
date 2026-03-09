import numpy as np

# ─── Reproducible setup ───
np.random.seed(42)

# ─── Generate a toy 3-class dataset ───
# Three overlapping clusters — deliberately hard so the teacher's
# soft targets carry meaningful inter-class information
def make_data(n_per_class=200):
    X, y = [], []
    centers = [(-1.5, -1.0), (1.5, -1.0), (0.0, 1.5)]
    for cls, (cx, cy) in enumerate(centers):
        pts = np.random.randn(n_per_class, 2) * 0.9 + [cx, cy]
        X.append(pts)
        y.append(np.full(n_per_class, cls))
    return np.vstack(X), np.concatenate(y).astype(int)

X_train, y_train = make_data(200)  # 600 samples
X_test, y_test = make_data(100)    # 300 samples

def one_hot(labels, num_classes=3):
    oh = np.zeros((len(labels), num_classes))
    oh[np.arange(len(labels)), labels] = 1.0
    return oh

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# ─── Simple MLP helper ───
def relu(x):
    return np.maximum(0, x)

def softmax_batch(logits, T=1.0):
    scaled = logits / T
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

class MLP:
    """Minimal 2-layer MLP: input → hidden (ReLU) → output (logits)."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * scale2
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        self.h_pre = X @ self.W1 + self.b1       # (batch, hidden)
        self.h = relu(self.h_pre)                 # (batch, hidden)
        self.logits = self.h @ self.W2 + self.b2  # (batch, output)
        return self.logits

    def backward(self, X, dlogits, lr=0.01):
        batch = X.shape[0]
        # Output layer gradients
        dW2 = self.h.T @ dlogits / batch
        db2 = np.mean(dlogits, axis=0)
        # Hidden layer gradients
        dh = dlogits @ self.W2.T
        dh_pre = dh * (self.h_pre > 0).astype(float)  # ReLU backward
        dW1 = X.T @ dh_pre / batch
        db1 = np.mean(dh_pre, axis=0)
        # SGD update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

def accuracy(model, X, y):
    logits = model.forward(X)
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

# ─── Train the teacher (big model: 128 hidden units) ───
teacher = MLP(2, 128, 3)
for epoch in range(300):
    logits = teacher.forward(X_train)
    probs = softmax_batch(logits, T=1.0)
    dlogits = probs - y_train_oh                # per-sample CE gradient
    teacher.backward(X_train, dlogits, lr=0.1)

teacher_acc = accuracy(teacher, X_test, y_test)
print(f"Teacher accuracy: {teacher_acc:.4f}")
# Teacher accuracy: 0.8833  (results vary by random seed)

# ─── Approach 1: Student trained from scratch (hard labels only) ───
student_scratch = MLP(2, 32, 3)
scratch_history = []
for epoch in range(300):
    logits = student_scratch.forward(X_train)
    probs = softmax_batch(logits, T=1.0)
    dlogits = probs - y_train_oh                # per-sample CE gradient
    student_scratch.backward(X_train, dlogits, lr=0.1)
    if epoch % 10 == 0:
        scratch_history.append(accuracy(student_scratch, X_test, y_test))

print(f"Student (scratch):      {accuracy(student_scratch, X_test, y_test):.4f}")

# ─── Approach 2: Student trained on teacher's hard outputs (SFT-style) ───
# Use teacher's argmax as labels — no soft targets
teacher_hard = np.argmax(teacher.forward(X_train), axis=1)
teacher_hard_oh = one_hot(teacher_hard)

student_hard = MLP(2, 32, 3)
hard_history = []
for epoch in range(300):
    logits = student_hard.forward(X_train)
    probs = softmax_batch(logits, T=1.0)
    dlogits = probs - teacher_hard_oh            # per-sample CE gradient
    student_hard.backward(X_train, dlogits, lr=0.1)
    if epoch % 10 == 0:
        hard_history.append(accuracy(student_hard, X_test, y_test))

print(f"Student (hard distill): {accuracy(student_hard, X_test, y_test):.4f}")

# ─── Approach 3: Full knowledge distillation (soft targets, T=4, α=0.9) ───
T = 4.0
alpha = 0.9
teacher_logits_all = teacher.forward(X_train)  # (600, 3)

student_kd = MLP(2, 32, 3)
kd_history = []
for epoch in range(300):
    student_logits = student_kd.forward(X_train)

    # Soft-target gradient: T² · d/dz KL(p_teacher^T || p_student^T)
    p_teacher_soft = softmax_batch(teacher_logits_all, T=T)
    p_student_soft = softmax_batch(student_logits, T=T)
    soft_grad = T * (p_student_soft - p_teacher_soft)  # T² absorbed: T·(1/T)·(...) = (...)
    # Wait — the 1/T from the softmax derivative and T² give us just T × (p_s - p_t)

    # Hard-label gradient: d/dz CE(y, p_student)
    p_student_hard = softmax_batch(student_logits, T=1.0)
    hard_grad = (p_student_hard - y_train_oh)

    # Combined gradient (per-sample, same format as approaches 1 and 2)
    dlogits = alpha * soft_grad + (1 - alpha) * hard_grad
    student_kd.backward(X_train, dlogits, lr=0.1)

    if epoch % 10 == 0:
        kd_history.append(accuracy(student_kd, X_test, y_test))

print(f"Student (KD, T=4):      {accuracy(student_kd, X_test, y_test):.4f}")

# Teacher accuracy:       0.8833  (results vary by random seed)
# Student (scratch):      0.8433
# Student (hard distill):  0.8500
# Student (KD, T=4):      0.8600
