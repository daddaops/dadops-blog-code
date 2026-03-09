import numpy as np


class KernelSVM:
    """SVM with kernel trick using simplified SMO-style coordinate descent."""

    def __init__(self, kernel='rbf', C=1.0, gamma=1.0, degree=3, epochs=200):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.epochs = epochs

    def _kernel(self, x1, x2):
        if self.kernel == 'linear':
            return x1 @ x2.T
        elif self.kernel == 'poly':
            return (x1 @ x2.T + 1) ** self.degree
        elif self.kernel == 'rbf':
            # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 * x1 . x2
            sq1 = np.sum(x1 ** 2, axis=1, keepdims=True)
            sq2 = np.sum(x2 ** 2, axis=1, keepdims=True)
            dist_sq = sq1 + sq2.T - 2 * (x1 @ x2.T)
            return np.exp(-self.gamma * dist_sq)

    def fit(self, X, y):
        n = len(y)
        self.X_train = X
        self.y_train = y
        self.alphas = np.zeros(n)
        self.b = 0.0
        K = self._kernel(X, X)

        for epoch in range(self.epochs):
            for i in range(n):
                # decision value for point i
                f_i = np.sum(self.alphas * y * K[i]) + self.b
                error_i = f_i - y[i]

                # KKT violation check
                if (y[i] * f_i < 1 and self.alphas[i] < self.C) or \
                   (y[i] * f_i > 1 and self.alphas[i] > 0):
                    # pick a random j != i
                    j = i
                    while j == i:
                        j = np.random.randint(n)

                    f_j = np.sum(self.alphas * y * K[j]) + self.b
                    error_j = f_j - y[j]

                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if eta <= 0:
                        continue

                    alpha_j_old = self.alphas[j]
                    alpha_i_old = self.alphas[i]

                    # compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L == H:
                        continue

                    # update alpha_j
                    self.alphas[j] += y[j] * (error_i - error_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)

                    # update alpha_i
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])

                    # update bias
                    b1 = self.b - error_i - y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - error_j - y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

    def decision_function(self, X):
        K = self._kernel(X, self.X_train)
        return K @ (self.alphas * self.y_train) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


if __name__ == "__main__":
    # Concentric circles dataset
    np.random.seed(42)
    n_per_class = 60
    angles = np.random.uniform(0, 2 * np.pi, n_per_class)
    r_inner = 1.0 + np.random.randn(n_per_class) * 0.15
    r_outer = 3.0 + np.random.randn(n_per_class) * 0.3
    X_inner = np.column_stack([r_inner * np.cos(angles), r_inner * np.sin(angles)])
    X_outer = np.column_stack([r_outer * np.cos(angles), r_outer * np.sin(angles)])
    X_circles = np.vstack([X_inner, X_outer])
    y_circles = np.array([1]*n_per_class + [-1]*n_per_class)

    for kernel_name in ['linear', 'poly', 'rbf']:
        svm = KernelSVM(kernel=kernel_name, C=10.0, gamma=1.0, degree=2, epochs=100)
        svm.fit(X_circles, y_circles)
        preds = svm.predict(X_circles)
        acc = np.mean(preds == y_circles) * 100
        sv_count = np.sum(svm.alphas > 1e-5)
        print(f"{kernel_name:>6} kernel: accuracy={acc:.0f}%, support vectors={sv_count}")
