import numpy as np
from logistic_core import sigmoid, LogisticRegression

# Comparison: Logistic Regression vs SVM vs Naive Bayes vs Decision Tree
# (using our from-scratch implementations where possible)

results = {
    "Logistic Regression": {"train_speed": "fast (SGD)", "calibration": "excellent",
        "interpretability": "high (weights = feature importance)",
        "online": True, "best_for": "first baseline, probabilities needed"},
    "SVM (RBF)": {"train_speed": "slow (O(n^2))", "calibration": "poor",
        "interpretability": "low (kernel space)",
        "online": False, "best_for": "small data, clear margins"},
    "Naive Bayes": {"train_speed": "fastest (single pass)", "calibration": "poor",
        "interpretability": "moderate (per-class feature probs)",
        "online": True, "best_for": "text, very small data, multi-class"},
    "Decision Tree": {"train_speed": "moderate", "calibration": "poor",
        "interpretability": "highest (readable rules)",
        "online": False, "best_for": "mixed feature types, interpretability"}
}

print(f"{'Model':<25} {'Calibration':<14} {'Online?':<10} {'Speed'}")
print("-" * 70)
for name, props in results.items():
    online = "Yes" if props["online"] else "No"
    print(f"{name:<25} {props['calibration']:<14} {online:<10} {props['train_speed']}")

# Train a model on XOR to show weight interpretability
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y_xor = np.array([0, 1, 1, 0], dtype=float)
lr_model = LogisticRegression(2)
lr_model.fit(X_xor, y_xor, lr=1.0, epochs=1000)

# Key insight: LR coefficients directly show feature importance
# Positive weight = feature pushes toward class 1
# Negative weight = feature pushes toward class 0
# |weight| = strength of influence
print(f"\nLR weights reveal WHY: w = {lr_model.w.round(3)}")
print("Each coefficient tells you the direction AND magnitude of influence")
