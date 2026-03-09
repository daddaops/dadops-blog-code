import numpy as np

def focal_loss(p_true, gamma=2.0, alpha=0.25):
    """Focal loss: upweights hard examples, downweights easy ones.

    p_true: model's predicted probability for the correct class
    gamma: focusing parameter (0 = standard cross-entropy)
    alpha: class-balancing weight
    """
    return -alpha * (1 - p_true) ** gamma * np.log(p_true + 1e-8)

# Compare focal loss for easy vs hard examples
p_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # model confidence
for gamma in [0, 1, 2, 5]:
    losses = focal_loss(p_values, gamma=gamma)
    weights = (1 - p_values) ** gamma  # the modulating factor alone
    print(f"gamma={gamma}: weights = {np.round(weights, 4)}")
# gamma=0: weights = [1.     1.     1.     1.     1.    ]  (standard CE)
# gamma=1: weights = [0.9    0.7    0.5    0.3    0.1   ]  (mild focusing)
# gamma=2: weights = [0.81   0.49   0.25   0.09   0.01  ]  (strong focusing)
# gamma=5: weights = [0.5905 0.1681 0.0312 0.0024 0.0   ]  (extreme)
