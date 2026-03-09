"""
DPO vs RLHF Comparison

Trains both DPO and reward-weighted RLHF on the same toy problem
and compares their preference accuracy and training stability.

Blog post: https://dadops.dev/blog/dpo-from-scratch/
"""
import numpy as np
from dpo_training import sigmoid, ToyLM, make_preference_data, train_dpo, evaluate_preferences


def train_rlhf_ppo(model, ref_model, reward_fn, n_prompts=10,
                    n_responses=5, beta=0.1, lr=0.005, epochs=80):
    """
    Reward-weighted RLHF training.

    Uses reward-weighted regression: computes advantages for all responses,
    converts to a target distribution, and minimizes cross-entropy.
    This is more stable than single-sample REINFORCE for toy problems.
    """
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in range(n_prompts):
            h = np.tanh(model.W1[x])
            logits = h @ model.W2
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()

            # Compute reward and KL for all responses
            rewards = np.array([reward_fn(x, y) for y in range(n_responses)])
            ref_lps = np.array([ref_model.log_probs(x, y) for y in range(n_responses)])
            cur_lps = np.array([model.log_probs(x, y) for y in range(n_responses)])
            kl_penalties = beta * (cur_lps - ref_lps)
            advantages = rewards - kl_penalties

            # Softmax advantages to get target distribution
            adv_temp = advantages / max(advantages.std(), 0.1)
            target_probs = np.exp(adv_temp - adv_temp.max())
            target_probs /= target_probs.sum()

            # Cross-entropy gradient: push toward target distribution
            grad_logits = probs - target_probs

            d_W2 = np.outer(h, grad_logits)
            dh = 1.0 - h ** 2
            d_W1_row = (grad_logits @ model.W2.T) * dh

            model.W2 -= lr * d_W2
            model.W1[x] -= lr * d_W1_row

            epoch_loss += np.sum(probs * kl_penalties) - np.sum(probs * rewards)

        losses.append(epoch_loss / n_prompts)
    return losses


def evaluate_prob_accuracy(model, pairs):
    """Check if model assigns higher probability to preferred responses."""
    correct = 0
    for x, y_w, y_l in pairs:
        lp_w = model.log_probs(x, y_w)
        lp_l = model.log_probs(x, y_l)
        if lp_w > lp_l:
            correct += 1
    return correct / len(pairs)


if __name__ == "__main__":
    np.random.seed(42)
    pairs, true_q = make_preference_data(n_prompts=10, n_responses=5, n_pairs=200)

    # DPO training
    dpo_model = ToyLM(10, 5)
    dpo_ref = dpo_model.copy()
    dpo_losses = train_dpo(dpo_model, dpo_ref, pairs, beta=0.1, lr=0.05, epochs=80)

    # RLHF training (needs an oracle reward function)
    rlhf_model = ToyLM(10, 5)
    rlhf_model.W1 = dpo_ref.W1.copy()  # Same initialization
    rlhf_model.W2 = dpo_ref.W2.copy()
    rlhf_ref = rlhf_model.copy()
    reward_fn = lambda x, y: true_q[x, y]  # Oracle reward for fair comparison
    rlhf_losses = train_rlhf_ppo(rlhf_model, rlhf_ref, reward_fn,
                                   beta=0.1, lr=0.02, epochs=80)

    # Evaluate both methods
    dpo_acc = evaluate_preferences(dpo_model, dpo_ref, pairs)
    dpo_prob_acc = evaluate_prob_accuracy(dpo_model, pairs)
    rlhf_prob_acc = evaluate_prob_accuracy(rlhf_model, pairs)

    print("DPO vs RLHF Comparison")
    print("=" * 55)
    print(f"DPO  accuracy: {dpo_acc:.1%}  | Final loss: {dpo_losses[-1]:.3f}")
    print(f"RLHF accuracy: {rlhf_prob_acc:.1%}  | Final loss: {rlhf_losses[-1]:.3f}")
    print()
    print("Loss trajectory (every 10 epochs):")
    print(f"{'Epoch':>6s}  {'DPO':>8s}  {'RLHF':>8s}")
    for i in range(0, 80, 10):
        print(f"{i:>6d}  {dpo_losses[i]:>8.3f}  {rlhf_losses[i]:>8.3f}")
