"""Complete speculative decoding loop with toy models for demonstration."""

import numpy as np


def sample(distribution):
    """Sample a token index from a probability distribution."""
    return np.random.choice(len(distribution), p=distribution)


class ToyDraftModel:
    """Toy draft model with a simple bigram-like distribution."""
    def __init__(self, vocab_size=8):
        self.vocab_size = vocab_size
        # Pre-generate some transition distributions
        np.random.seed(123)
        self.transitions = {
            i: np.random.dirichlet(np.ones(vocab_size) * 2)
            for i in range(vocab_size)
        }

    def predict(self, context):
        last = context[-1] if context else 0
        return self.transitions[last % self.vocab_size]

    def rollback_cache(self, length):
        pass  # No-op for toy model


class ToyTargetModel:
    """Toy target model — similar but slightly different distributions."""
    def __init__(self, vocab_size=8):
        self.vocab_size = vocab_size
        np.random.seed(456)
        self.transitions = {
            i: np.random.dirichlet(np.ones(vocab_size) * 2)
            for i in range(vocab_size)
        }

    def verify(self, prefix, draft_tokens):
        """Return K+1 distributions for verification."""
        dists = []
        ctx = list(prefix)
        for t in draft_tokens:
            last = ctx[-1] if ctx else 0
            dists.append(self.transitions[last % self.vocab_size])
            ctx.append(t)
        # Bonus position
        last = ctx[-1] if ctx else 0
        dists.append(self.transitions[last % self.vocab_size])
        return dists


def speculative_decode(prompt_tokens, draft_model, target_model,
                       K=5, max_new_tokens=100):
    """
    Complete speculative decoding loop.

    draft_model.predict(context) -> probability distribution over vocab
    target_model.verify(prefix, draft_tokens) -> list of (K+1) distributions
    """
    generated = list(prompt_tokens)

    while len(generated) - len(prompt_tokens) < max_new_tokens:
        start = len(generated)

        # -- Phase 1: Draft K tokens autoregressively (cheap) --
        draft_tokens = []
        draft_dists = []
        ctx = list(generated)
        for _ in range(K):
            q = draft_model.predict(ctx)
            token = sample(q)
            draft_tokens.append(token)
            draft_dists.append(q)
            ctx.append(token)

        # -- Phase 2: Verify all K in ONE target model pass --
        target_dists = target_model.verify(generated, draft_tokens)
        # target_dists[i] = target's distribution at position (start + i)
        # We get K+1 distributions: positions 0..K

        # -- Phase 3: Accept/reject with rejection sampling --
        n_accepted = 0
        for i in range(K):
            p = target_dists[i]
            q = draft_dists[i]
            token = draft_tokens[i]

            accept_prob = min(1.0, p[token] / q[token])
            if np.random.random() < accept_prob:
                generated.append(token)         # accept draft
                n_accepted += 1
            else:
                # Reject -- sample correction from residual
                residual = np.maximum(0, p - q)
                residual /= residual.sum()
                correction = np.random.choice(len(p), p=residual)
                generated.append(correction)    # append correction
                break                           # stop verifying

        # -- Bonus: if ALL K accepted, free extra token --
        if n_accepted == K:
            bonus = sample(target_dists[K])
            generated.append(bonus)             # K+1 tokens this step!

        # Roll back draft model's KV cache to match accepted length
        draft_model.rollback_cache(start + n_accepted + 1)

    return generated


if __name__ == "__main__":
    np.random.seed(42)

    draft = ToyDraftModel(vocab_size=8)
    target = ToyTargetModel(vocab_size=8)
    prompt = [0, 1, 2]

    result = speculative_decode(prompt, draft, target, K=5, max_new_tokens=30)

    print(f"Prompt tokens:    {prompt}")
    print(f"Generated tokens: {result[len(prompt):]}")
    print(f"Total length:     {len(result)} (prompt={len(prompt)}, "
          f"generated={len(result)-len(prompt)})")
