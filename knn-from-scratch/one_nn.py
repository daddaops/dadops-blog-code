import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class OneNearestNeighbor:
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_one(self, x):
        distances = [euclidean_distance(x, xt) for xt in self.X_train]
        nearest_idx = np.argmin(distances)
        return self.y_train[nearest_idx]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

def make_data():
    # Generate two spiraling clusters
    np.random.seed(42)
    n = 100
    t = np.linspace(0, 4 * np.pi, n)
    X_class0 = np.column_stack([t * np.cos(t), t * np.sin(t)]) + np.random.randn(n, 2) * 0.5
    X_class1 = np.column_stack([t * np.cos(t + np.pi), t * np.sin(t + np.pi)]) + np.random.randn(n, 2) * 0.5
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * n + [1] * n)
    return X, y

if __name__ == "__main__":
    X, y = make_data()

    model = OneNearestNeighbor()
    model.fit(X, y)

    # 1-NN has PERFECT training accuracy (it memorizes everything)
    train_preds = model.predict(X)
    print(f"Training accuracy: {np.mean(train_preds == y):.2f}")  # 1.00
