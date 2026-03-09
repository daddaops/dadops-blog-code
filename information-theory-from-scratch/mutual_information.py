import numpy as np

def mutual_information(x, y, bins=20):
    """Estimate MI between continuous x and discrete y using binning."""
    x_binned = np.digitize(x, np.linspace(x.min(), x.max(), bins))

    # Joint and marginal distributions
    joint = np.zeros((bins + 1, int(y.max()) + 1))
    for xi, yi in zip(x_binned, y.astype(int)):
        joint[xi, yi] += 1
    joint /= joint.sum()

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mask = joint > 0
    independent = px * py
    mi = np.sum(joint[mask] * np.log(joint[mask] / independent[mask]))
    return mi

# --- Feature selection on synthetic data ---
rng = np.random.default_rng(42)
n = 2000

# Feature 0: strongly predictive of label
x0 = rng.normal(0, 1, n)
y = (x0 > 0).astype(int)  # label determined by x0

# Feature 1: weakly predictive (noisy copy)
x1 = x0 + rng.normal(0, 3, n)

# Feature 2: pure noise (independent of label)
x2 = rng.normal(0, 1, n)

features = [x0, x1, x2]
names = ["x0 (strong)", "x1 (weak)", "x2 (noise)"]

print("Mutual Information with target label:")
for name, feat in zip(names, features):
    mi = mutual_information(feat, y)
    print(f"  {name}: MI = {mi:.4f} nats")
# x0 has highest MI — it determines the label
# x2 has near-zero MI — it's independent noise
