"""
DPO vs RLHF/PPO Comparison

Trains both DPO and simplified PPO-style RLHF on the same toy problem
and compares their preference accuracy and training stability.

Blog post: https://dadops.dev/blog/dpo-from-scratch/
"""
import numpy as np
from dpo_training import sigmoid, ToyLM, make_preference_data, train_dpo, evaluate_preferences


def train_rlhf_ppo(model, ref_model, reward_fn, n_prompts=10,
                    n_responses=5, beta=0.1, lr=0.005, epochs=80):
    """Simplified PPO-style RLHF training for comparison."""
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in range(n_prompts):
            h = np.tanh(model.W1[x])
            logits = h @ model.W2
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            y = np.random.choice(n_responses, p=probs)

            reward = reward_fn(x, y)
            ref_lp = ref_model.log_probs(x, y)
            cur_lp = model.log_probs(x, y)
            kl_penalty = beta * (cur_lp - ref_lp)
            advantage = reward - kl_penalty

            d_logits = -probs.copy()
            d_logits[y] += 1.0
            noise = np.random.randn() * 0.3
            grad = advantage * d_logits * (1.0 + noise)

            d_W2 = np.outer(h, grad)
            model.W2 -= lr * d_W2

            epoch_loss += -advantage

        losses.append(epoch_loss / n_prompts)
    return losses


if __name__ == "__main__":
    np.random.seed(42)
    pairs, true_q = make_preference_data(n_prompts=10, n_responses=5, n_pairs=200)

    # DPO training
    dpo_model = ToyLM(10, 5)
    dpo_ref = dpo_model.copy()
    dpo_losses = train_dpo(dpo_model, dpo_ref, pairs, beta=0.1, lr=0.005, epochs=80)

    # RLHF/PPO training (needs an oracle reward function)
    rlhf_model = ToyLM(10, 5)
    rlhf_model.W1 = dpo_ref.W1.copy()
    rlhf_model.W2 = dpo_ref.W2.copy()
    rlhf_ref = rlhf_model.copy()
    reward_fn = lambda x, y: true_q[x, y]
    rlhf_losses = train_rlhf_ppo(rlhf_model, rlhf_ref, reward_fn,
                                   beta=0.1, lr=0.005, epochs=80)

    dpo_acc = evaluate_preferences(dpo_model, dpo_ref, pairs)
    rlhf_acc = evaluate_preferences(rlhf_model, rlhf_ref, pairs)

    print("DPO vs RLHF/PPO Comparison")
    print("=" * 55)
    print(f"DPO  accuracy: {dpo_acc:.1%}  | Final loss: {dpo_losses[-1]:.3f}")
    print(f"RLHF accuracy: {rlhf_acc:.1%}  | Final loss: {rlhf_losses[-1]:.3f}")
    print()
    print("Loss trajectory (every 10 epochs):")
    print(f"{'Epoch':>6s}  {'DPO':>8s}  {'RLHF':>8s}")
    for i in range(0, 80, 10):
        print(f"{i:>6d}  {dpo_losses[i]:>8.3f}  {rlhf_losses[i]:>8.3f}")
