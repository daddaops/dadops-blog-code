import numpy as np
from one_nn import euclidean_distance

class KNNRegressor:
    def __init__(self, k=5, weighted=False):
        self.k = k
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_one(self, x):
        distances = np.array([euclidean_distance(x, xt) for xt in self.X_train])
        k_nearest = np.argsort(distances)[:self.k]
        k_dists = distances[k_nearest]
        k_targets = self.y_train[k_nearest]

        if self.weighted and np.any(k_dists > 0):
            weights = 1.0 / (k_dists + 1e-10)
            return np.average(k_targets, weights=weights)
        return np.mean(k_targets)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

# True function: sin wave with noise
np.random.seed(42)
X_train = np.random.uniform(0, 2 * np.pi, 40).reshape(-1, 1)
y_train = np.sin(X_train.ravel()) + np.random.randn(40) * 0.2

X_test = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)

for k, weighted in [(1, False), (5, False), (5, True), (20, False)]:
    model = KNNRegressor(k=k, weighted=weighted)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = np.mean((preds - np.sin(X_test.ravel())) ** 2)
    label = f"k={k}, {'weighted' if weighted else 'uniform':<8}"
    print(f"{label}  MSE={mse:.4f}")
# k=1, uniform   MSE=0.0298  (interpolates every point)
# k=5, uniform   MSE=0.0139  (smooth, good approximation)
# k=5, weighted   MSE=0.0118  (smoother, closer neighbors dominate)
# k=20, uniform  MSE=0.0933  (over-smoothed, loses the wave)
