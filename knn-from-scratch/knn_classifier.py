import numpy as np
from one_nn import euclidean_distance, make_data

class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

    def predict_one(self, x):
        distances = np.array([euclidean_distance(x, xt) for xt in self.X_train])
        k_nearest = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_nearest]
        # Majority vote
        counts = [np.sum(k_labels == c) for c in self.classes]
        return self.classes[np.argmax(counts)]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

X, y = make_data()

# Evaluate k=1 through k=50 with train/test split
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.7 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

for k in [1, 5, 15, 50]:
    model = KNearestNeighbors(k=k)
    model.fit(X_train, y_train)
    train_acc = np.mean(model.predict(X_train) == y_train)
    test_acc = np.mean(model.predict(X_test) == y_test)
    print(f"k={k:<3}  train={train_acc:.3f}  test={test_acc:.3f}")
# k=1   train=1.000  test=0.900
# k=5   train=0.914  test=0.750
# k=15  train=0.657  test=0.517
# k=50  train=0.450  test=0.400
