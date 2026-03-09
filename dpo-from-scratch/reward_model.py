"""
Bradley-Terry Reward Model Training

Implements the reward model training step of RLHF: learns to predict which
response humans prefer using the Bradley-Terry preference model. This is the
component that DPO eliminates.

Blog post: https://dadops.dev/blog/dpo-from-scratch/
"""
import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


class SimpleRewardModel:
    """A simple parameterized reward function for demonstration."""
    def __init__(self, n_prompts, n_responses):
        self.scores = np.random.randn(n_prompts, n_responses) * 0.01

    def __call__(self, x, y):
        return self.scores[x, y]

    def update(self, x, y, delta):
        self.scores[x, y] += delta


def train_reward_model(preferences, reward_fn, lr=0.01, steps=200):
    """
    Train a reward model on human preference pairs.

    preferences: list of (x, y_w, y_l) — prompt, preferred, rejected
    reward_fn:   callable(x, y) -> scalar, parameterized reward model
    """
    losses = []
    for step in range(steps):
        total_loss = 0.0
        for x, y_w, y_l in preferences:
            # Reward for preferred and rejected responses
            r_w = reward_fn(x, y_w)
            r_l = reward_fn(x, y_l)

            # Bradley-Terry probability: sigma(r_w - r_l)
            prob_prefer_w = sigmoid(r_w - r_l)

            # Negative log-likelihood loss
            loss = -np.log(prob_prefer_w + 1e-10)
            total_loss += loss

            # Gradient: d_loss/d_r_w = -(1 - prob), d_loss/d_r_l = prob - 1
            grad = -(1.0 - prob_prefer_w)
            reward_fn.update(x, y_w, lr * -grad)   # increase r_w
            reward_fn.update(x, y_l, lr * grad)     # decrease r_l

        losses.append(total_loss / len(preferences))
    return losses


if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic preference pairs
    n_prompts, n_responses = 10, 5
    true_quality = np.random.randn(n_prompts, n_responses)
    pairs = []
    for _ in range(200):
        x = np.random.randint(n_prompts)
        y1, y2 = np.random.choice(n_responses, 2, replace=False)
        if true_quality[x, y1] > true_quality[x, y2]:
            pairs.append((x, y1, y2))
        else:
            pairs.append((x, y2, y1))

    reward_model = SimpleRewardModel(n_prompts, n_responses)
    losses = train_reward_model(pairs, reward_model, lr=0.01, steps=200)

    print("Reward Model Training")
    print("=" * 50)
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss:   {losses[-1]:.4f}")
    print(f"Loss at step 50:  {losses[49]:.4f}")
    print(f"Loss at step 100: {losses[99]:.4f}")

    # Check accuracy
    correct = 0
    for x, y_w, y_l in pairs:
        if reward_model(x, y_w) > reward_model(x, y_l):
            correct += 1
    print(f"Preference accuracy: {correct/len(pairs):.1%}")
