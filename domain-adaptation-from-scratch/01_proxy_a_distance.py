import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def proxy_a_distance(source_features, target_features):
    """Estimate domain divergence via Proxy A-distance.

    Returns d_hat_A in [0, 2]. Higher means more domain shift.
    """
    X = np.vstack([source_features, target_features])
    y = np.concatenate([
        np.zeros(len(source_features)),  # source = 0
        np.ones(len(target_features))    # target = 1
    ])

    clf = LogisticRegression(max_iter=1000)
    cv_acc = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    err = 1.0 - cv_acc.mean()

    d_hat_a = 2.0 * (1.0 - 2.0 * err)
    return max(0.0, d_hat_a)  # clamp to [0, 2]

# Example: source and target are 50-dim features
source = np.random.randn(200, 50)
target = np.random.randn(200, 50) + 0.5  # shifted by 0.5
print(f"Proxy A-distance: {proxy_a_distance(source, target):.3f}")
# Higher values mean easier to distinguish = more domain shift
