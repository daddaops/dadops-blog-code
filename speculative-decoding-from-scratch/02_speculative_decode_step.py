"""Speculative decoding step pseudocode with toy models to demonstrate the flow."""

import numpy as np


def sample_from(distribution):
    """Sample a token index from a probability distribution."""
    return np.random.choice(len(distribution), p=distribution)


class ToyDraftModel:
    """Simple draft model that returns a fixed distribution."""
    def __init__(self, vocab_size=4):
        self.vocab_size = vocab_size

    def predict(self, context):
        # Return a simple distribution regardless of context
        q = np.random.dirichlet(np.ones(self.vocab_size))
        return q


class ToyTargetModel:
    """Simple target model that returns fixed distributions for verification."""
    def __init__(self, vocab_size=4):
        self.vocab_size = vocab_size

    def verify(self, prefix, draft_tokens):
        # Return K+1 distributions (one per draft position + bonus)
        return [np.random.dirichlet(np.ones(self.vocab_size))
                for _ in range(len(draft_tokens) + 1)]


def accept(target_dist, draft_dist, token):
    """Accept with probability min(1, p/q)."""
    accept_prob = min(1.0, target_dist[token] / draft_dist[token])
    return np.random.random() < accept_prob


def sample_correction(target_dist, draft_dist):
    """Sample from the residual distribution."""
    residual = np.maximum(0, target_dist - draft_dist)
    residual /= residual.sum()
    return np.random.choice(len(target_dist), p=residual)


def speculative_decode_step(prefix, draft_model, target_model, K):
    """One step: returns 1 to K+1 new tokens."""

    # -- Draft: K tokens from small model (cheap) --
    draft_tokens, draft_probs = [], []
    context = list(prefix)
    for _ in range(K):
        q = draft_model.predict(context)       # fast!
        token = sample_from(q)
        draft_tokens.append(token)
        draft_probs.append(q)
        context.append(token)

    # -- Verify: score all K positions in ONE target pass --
    # This is the magic -- parallel, like prefill
    target_dists = target_model.verify(prefix, draft_tokens)
    # target_dists[i] gives the target's distribution at position i

    # -- Accept/reject via rejection sampling --
    accepted = []
    for i in range(K):
        if accept(target_dists[i], draft_probs[i], draft_tokens[i]):
            accepted.append(draft_tokens[i])
        else:
            correction = sample_correction(target_dists[i], draft_probs[i])
            accepted.append(correction)
            return accepted               # stop at first rejection

    # All K accepted! Bonus token from position K+1
    bonus = sample_from(target_dists[K])
    accepted.append(bonus)
    return accepted                       # K+1 tokens!


if __name__ == "__main__":
    np.random.seed(42)
    draft = ToyDraftModel(vocab_size=4)
    target = ToyTargetModel(vocab_size=4)
    prefix = [0, 1, 2]  # some prompt tokens
    K = 5

    for trial in range(5):
        result = speculative_decode_step(prefix, draft, target, K)
        print(f"Trial {trial+1}: accepted {len(result)} tokens "
              f"(max possible: {K+1}) -> {result}")
