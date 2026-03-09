"""Attention mechanism for encoder-decoder ASR."""
import numpy as np


def attention_mechanism(encoder_states, decoder_query):
    """Compute attention weights and context vector.
    encoder_states: (T, D) hidden states from the encoder
    decoder_query:  (D,) current decoder hidden state
    Returns: attention weights (T,), context vector (D,)
    """
    # Dot-product attention scores
    scores = encoder_states @ decoder_query  # (T,)
    # Softmax to get attention weights
    scores -= np.max(scores)  # numerical stability
    weights = np.exp(scores) / np.sum(np.exp(scores))
    # Weighted sum of encoder states
    context = weights @ encoder_states  # (D,)
    return weights, context


if __name__ == "__main__":
    # Simulate encoder output for "set a timer" (T=75 after subsampling, D=256)
    np.random.seed(42)
    T_enc, D = 75, 256
    encoder_out = np.random.randn(T_enc, D) * 0.1

    # Make encoder states cluster by rough word position
    # "set" ~frames 0-15, "a" ~frames 20-25, "timer" ~frames 30-65
    for t in range(0, 15):  encoder_out[t, :10] += 2.0
    for t in range(20, 25): encoder_out[t, 10:20] += 2.0
    for t in range(30, 65): encoder_out[t, 20:30] += 2.0

    # Decoder generates one char at a time; simulate query for 't' in "timer"
    # Query should attend to frames 30-65
    decoder_query = np.zeros(D)
    decoder_query[20:30] = 2.0  # aligns with "timer" frames

    weights, context = attention_mechanism(encoder_out, decoder_query)

    peak = np.argmax(weights)
    print(f"Encoder states: {encoder_out.shape}")
    print(f"Attention weights sum: {weights.sum():.4f}")
    print(f"Peak attention at frame {peak} (expected: 30-65 range)")
    print(f"Context vector norm: {np.linalg.norm(context):.4f}")

    # Show alignment: which frames does decoder attend to?
    top5 = np.argsort(weights)[-5:][::-1]
    print(f"Top-5 attended frames: {top5}")
