"""
DPO Training from Scratch

Implements Direct Preference Optimization on a toy language model:
- ToyLM: a simple prompt→response model with learnable weights
- DPO loss computation with implicit reward formulation
- Full training loop with backpropagation through softmax
- Evaluation of learned preferences

Blog post: https://dadops.dev/blog/dpo-from-scratch/
"""
import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def dpo_loss(pi_theta_logprobs_w, pi_theta_logprobs_l,
             pi_ref_logprobs_w, pi_ref_logprobs_l, beta=0.1):
    """
    Compute the DPO loss for a batch of preference pairs.

    All inputs are log-probabilities of the full response sequence,
    i.e., sum of log p(token_t | tokens_<t) for each token.
    """
    log_ratio_w = pi_theta_logprobs_w - pi_ref_logprobs_w
    log_ratio_l = pi_theta_logprobs_l - pi_ref_logprobs_l
    logits = beta * (log_ratio_w - log_ratio_l)
    loss = -np.log(sigmoid(logits) + 1e-10)
    return np.mean(loss)


class ToyLM:
    """A tiny 'language model' that maps prompt IDs to response logits."""
    def __init__(self, n_prompts, n_responses, hidden=32):
        self.W1 = np.random.randn(n_prompts, hidden) * 0.1
        self.W2 = np.random.randn(hidden, n_responses) * 0.1

    def log_probs(self, prompt_id, response_id):
        """Compute log p(response | prompt)."""
        h = np.tanh(self.W1[prompt_id])
        logits = h @ self.W2
        max_logit = logits.max()
        log_p = logits - max_logit - np.log(np.sum(np.exp(logits - max_logit)))
        return log_p[response_id]

    def copy(self):
        clone = ToyLM.__new__(ToyLM)
        clone.W1 = self.W1.copy()
        clone.W2 = self.W2.copy()
        return clone


def make_preference_data(n_prompts=10, n_responses=5, n_pairs=100):
    """Generate synthetic preference pairs with a hidden 'true' quality."""
    true_quality = np.random.randn(n_prompts, n_responses)
    pairs = []
    for _ in range(n_pairs):
        x = np.random.randint(n_prompts)
        y1, y2 = np.random.choice(n_responses, 2, replace=False)
        if true_quality[x, y1] > true_quality[x, y2]:
            pairs.append((x, y1, y2))
        else:
            pairs.append((x, y2, y1))
    return pairs, true_quality


def train_dpo(model, ref_model, preferences, beta=0.1, lr=0.01, epochs=100):
    """Full DPO training loop."""
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        np.random.shuffle(preferences)
        for x, y_w, y_l in preferences:
            lp_w = model.log_probs(x, y_w)
            lp_l = model.log_probs(x, y_l)
            ref_lp_w = ref_model.log_probs(x, y_w)
            ref_lp_l = ref_model.log_probs(x, y_l)

            logit = beta * ((lp_w - ref_lp_w) - (lp_l - ref_lp_l))
            loss = -np.log(sigmoid(logit) + 1e-10)
            epoch_loss += loss

            d_logit = sigmoid(logit) - 1.0

            h_x = np.tanh(model.W1[x])
            dh_x = 1.0 - h_x ** 2

            logits = h_x @ model.W2
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()

            d_logits_w = -probs.copy(); d_logits_w[y_w] += 1.0
            d_logits_l = -probs.copy(); d_logits_l[y_l] += 1.0

            d_W2 = np.outer(h_x, beta * d_logit * (d_logits_w - d_logits_l))
            d_h = beta * d_logit * ((d_logits_w - d_logits_l) @ model.W2.T)
            d_W1_row = d_h * dh_x

            model.W2 -= lr * d_W2
            model.W1[x] -= lr * d_W1_row

        losses.append(epoch_loss / len(preferences))
    return losses


def evaluate_preferences(model, ref_model, pairs, beta=0.1):
    """Check what fraction of preferences the trained policy matches."""
    correct = 0
    for x, y_w, y_l in pairs:
        lp_w = model.log_probs(x, y_w)
        lp_l = model.log_probs(x, y_l)
        ref_lp_w = ref_model.log_probs(x, y_w)
        ref_lp_l = ref_model.log_probs(x, y_l)

        implicit_r_w = beta * (lp_w - ref_lp_w)
        implicit_r_l = beta * (lp_l - ref_lp_l)

        if implicit_r_w > implicit_r_l:
            correct += 1

    return correct / len(pairs)


if __name__ == "__main__":
    np.random.seed(42)
    pairs, true_q = make_preference_data(n_prompts=10, n_responses=5, n_pairs=200)
    model = ToyLM(10, 5)
    ref_model = model.copy()
    losses = train_dpo(model, ref_model, pairs, beta=0.1, lr=0.005, epochs=80)

    print("DPO Training Results")
    print("=" * 50)
    print(f"Loss: {losses[0]:.3f} -> {losses[-1]:.3f}")

    accuracy = evaluate_preferences(model, ref_model, pairs, beta=0.1)
    print(f"Preference accuracy: {accuracy:.1%}")

    # Test DPO loss function independently
    print("\nDPO Loss Function Test")
    print("-" * 50)
    # Random log-probs (should give loss ≈ 0.693)
    random_loss = dpo_loss(
        np.array([-2.0]), np.array([-2.0]),
        np.array([-2.0]), np.array([-2.0]),
        beta=0.1
    )
    print(f"Random (equal) log-probs -> loss: {random_loss:.3f} (expected ~0.693)")

    # Preferred much more likely under policy -> loss should be low
    good_loss = dpo_loss(
        np.array([-1.0]), np.array([-3.0]),
        np.array([-2.0]), np.array([-2.0]),
        beta=0.1
    )
    print(f"Preferred wins clearly  -> loss: {good_loss:.3f} (expected < 0.693)")
