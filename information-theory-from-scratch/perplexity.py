import numpy as np

def perplexity_from_logprobs(log_probs):
    """Perplexity from per-token log-probabilities (base e)."""
    avg_nll = -np.mean(log_probs)
    return np.exp(avg_nll)

def perplexity_from_loss(cross_entropy_loss):
    """Perplexity from average cross-entropy loss."""
    return np.exp(cross_entropy_loss)

# --- Simulate three language models on the same 10-token sequence ---
vocab_size = 50000

# Model 1: Random (uniform over vocabulary)
random_logprobs = np.full(10, np.log(1.0 / vocab_size))
ppl_random = perplexity_from_logprobs(random_logprobs)

# Model 2: Bigram (moderately concentrated predictions)
bigram_logprobs = np.log([0.005, 0.01, 0.008, 0.02, 0.003,
                          0.015, 0.007, 0.012, 0.009, 0.006])
ppl_bigram = perplexity_from_logprobs(bigram_logprobs)

# Model 3: Transformer (highly concentrated predictions)
transformer_logprobs = np.log([0.15, 0.08, 0.12, 0.20, 0.05,
                               0.10, 0.07, 0.18, 0.09, 0.06])
ppl_transformer = perplexity_from_logprobs(transformer_logprobs)

print("Perplexity comparison:")
print(f"  Random:      PPL = {ppl_random:,.0f}")
print(f"  Bigram:      PPL = {ppl_bigram:,.0f}")
print(f"  Transformer: PPL = {ppl_transformer:.1f}")

# --- Connection: loss and perplexity are interchangeable ---
loss = 3.0
print(f"\nCross-entropy loss {loss:.1f} = PPL {perplexity_from_loss(loss):.1f}")
loss = 1.5
print(f"Cross-entropy loss {loss:.1f} = PPL {perplexity_from_loss(loss):.1f}")
