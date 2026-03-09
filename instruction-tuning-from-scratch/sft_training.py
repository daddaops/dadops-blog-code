import numpy as np

def sft_training_step(model_logits, token_ids, response_start_idx, vocab_size):
    """One step of supervised fine-tuning with instruction masking.

    model_logits: (seq_len, vocab_size) -- model predictions
    token_ids:    (seq_len,) -- ground truth token IDs
    response_start_idx: where the response begins (mask everything before)
    """
    # Cross-entropy loss only on response tokens
    total_loss = 0.0
    count = 0

    for t in range(response_start_idx, len(token_ids) - 1):
        # Softmax over vocabulary
        logits = model_logits[t]
        logits = logits - np.max(logits)  # numerical stability
        probs = np.exp(logits) / np.sum(np.exp(logits))

        # Cross-entropy: -log(probability of correct token)
        target = token_ids[t + 1]
        total_loss += -np.log(probs[target] + 1e-10)
        count += 1

    return total_loss / max(count, 1)

# Demonstrate on a toy example
np.random.seed(42)
seq_len, vocab_size = 20, 100
logits = np.random.randn(seq_len, vocab_size)
tokens = np.random.randint(0, vocab_size, seq_len)

# Instruction is tokens 0-9, response is tokens 10-19
loss_full = sft_training_step(logits, tokens, 0, vocab_size)
loss_masked = sft_training_step(logits, tokens, 10, vocab_size)

print(f"Loss on full sequence:      {loss_full:.4f}")
print(f"Loss on response only:      {loss_masked:.4f}")
print(f"Tokens used for loss (full): {len(tokens) - 1}")
print(f"Tokens used for loss (masked): {len(tokens) - 1 - 10}")
print(f"\nThe masked loss only optimizes response generation.")
print(f"Instruction tokens contribute to attention but not to gradients.")
# Loss on full sequence:      4.6257
# Loss on response only:      4.6506
# Tokens used for loss (full): 19
# Tokens used for loss (masked): 9
