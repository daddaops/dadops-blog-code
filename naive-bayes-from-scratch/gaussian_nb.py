"""Gaussian Naive Bayes classifier from scratch.

Assumes features are normally distributed within each class.
Uses log-space computation to avoid numerical underflow.
"""
import numpy as np

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = {}  # {class: (means, variances, prior)}
        for c in self.classes:
            X_c = X[y == c]
            self.params[c] = (
                X_c.mean(axis=0),            # mean of each feature
                X_c.var(axis=0) + 1e-9,      # variance (+ epsilon for stability)
                len(X_c) / len(X)            # class prior P(class)
            )
        return self

    def _log_gaussian(self, x, mean, var):
        """Log of Gaussian PDF: avoid underflow by staying in log-space."""
        return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)

    def predict(self, X):
        predictions = []
        for x in X:
            log_posteriors = []
            for c in self.classes:
                mean, var, prior = self.params[c]
                # log P(class) + sum of log P(feature_i | class)
                log_post = np.log(prior) + np.sum(self._log_gaussian(x, mean, var))
                log_posteriors.append(log_post)
            predictions.append(self.classes[np.argmax(log_posteriors)])
        return np.array(predictions)


if __name__ == "__main__":
    # Example: classify 2D data
    np.random.seed(42)
    X_train = np.vstack([
        np.random.randn(50, 2) + [2, 2],    # class 0: centered at (2,2)
        np.random.randn(50, 2) + [-1, -1],  # class 1: centered at (-1,-1)
    ])
    y_train = np.array([0]*50 + [1]*50)

    nb = GaussianNB().fit(X_train, y_train)
    print(nb.predict(np.array([[0, 0], [3, 3], [-2, -2]])))
