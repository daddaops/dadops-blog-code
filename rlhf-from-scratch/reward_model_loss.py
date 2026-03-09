"""Bradley-Terry preference loss for reward model training."""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def reward_model_loss(r_chosen, r_rejected):
    """
    Bradley-Terry preference loss.
    Pushes r_chosen above r_rejected.
    """
    diff = r_chosen - r_rejected
    loss = -np.log(sigmoid(diff))
    return loss


# Verify the worked example from the blog:
# r_chosen = 2.1, r_rejected = -0.8
# P(chosen > rejected) = sigmoid(2.1 - (-0.8)) = sigmoid(2.9) = 0.948
# Loss = -log(0.948) = 0.053
r_chosen = 2.1
r_rejected = -0.8
diff = r_chosen - r_rejected
prob = sigmoid(diff)
loss = reward_model_loss(r_chosen, r_rejected)

print(f"r_chosen={r_chosen}, r_rejected={r_rejected}")
print(f"diff = {diff:.1f}")
print(f"P(chosen > rejected) = sigmoid({diff:.1f}) = {prob:.3f}")
print(f"Loss = -log({prob:.3f}) = {loss:.3f}")
