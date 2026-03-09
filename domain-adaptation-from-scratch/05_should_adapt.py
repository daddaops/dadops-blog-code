import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def should_adapt(source_features, target_features, source_labels,
                 target_labels_estimate=None):
    """Heuristic check: is domain adaptation likely to help?

    Returns a dict with proxy A-distance and recommendation.
    """
    # 1. Measure domain gap via Proxy A-distance
    X = np.vstack([source_features, target_features])
    y_domain = np.concatenate([
        np.zeros(len(source_features)),
        np.ones(len(target_features))
    ])
    clf = LogisticRegression(max_iter=1000)
    err = 1.0 - cross_val_score(clf, X, y_domain, cv=5).mean()
    pad = max(0.0, 2.0 * (1.0 - 2.0 * err))

    # 2. Check label distribution similarity (if estimates available)
    label_warning = False
    if target_labels_estimate is not None:
        source_dist = np.bincount(source_labels) / len(source_labels)
        target_dist = np.bincount(target_labels_estimate) / len(target_labels_estimate)
        min_len = min(len(source_dist), len(target_dist))
        kl_approx = np.sum(source_dist[:min_len] *
                           np.log(source_dist[:min_len] /
                                  (target_dist[:min_len] + 1e-10) + 1e-10))
        label_warning = kl_approx > 0.5

    # 3. Recommendation
    if pad < 0.3:
        rec = "Low shift. Standard transfer likely sufficient."
    elif label_warning:
        rec = "Shift detected but label distributions differ. Adapt cautiously."
    else:
        rec = "Significant shift with similar labels. Adaptation recommended."

    return {"proxy_a_distance": pad, "recommendation": rec}

# --- Demo ---
np.random.seed(0)
source = np.random.randn(200, 20)
target = np.random.randn(200, 20) + 0.8
source_labels = np.random.randint(0, 3, 200)
result = should_adapt(source, target, source_labels)
print(f"PAD: {result['proxy_a_distance']:.3f}")
print(f"Recommendation: {result['recommendation']}")
