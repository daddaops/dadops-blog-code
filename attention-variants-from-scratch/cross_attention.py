"""Cross-Attention: Bridging Two Sequences.

Takes queries from one sequence (decoder) and keys/values from another
(encoder). Used in translation, vision-language, and image generation models.
"""
import numpy as np

def cross_attention(x_decoder, x_encoder, d_head):
    """Cross-attention: queries from decoder, keys/values from encoder."""
    d_model = x_decoder.shape[1]
    W_Q = np.random.randn(d_model, d_head) * 0.1
    W_K = np.random.randn(d_model, d_head) * 0.1
    W_V = np.random.randn(d_model, d_head) * 0.1

    Q = x_decoder @ W_Q    # (dec_len, d_head) — what am I looking for?
    K = x_encoder @ W_K    # (enc_len, d_head) — what do I contain?
    V = x_encoder @ W_V    # (enc_len, d_head) — what information do I carry?

    # Decoder tokens attend to encoder tokens
    scores = Q @ K.T / np.sqrt(d_head)    # (dec_len, enc_len)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    output = weights @ V                   # (dec_len, d_head)

    return output, weights

if __name__ == "__main__":
    # Simulate translation: French encoder -> English decoder
    np.random.seed(42)
    enc_len, dec_len, d_model, d_head = 5, 4, 32, 8
    x_encoder = np.random.randn(enc_len, d_model)  # "Le chat dort"
    x_decoder = np.random.randn(dec_len, d_model)  # "The cat sleeps"

    output, weights = cross_attention(x_decoder, x_encoder, d_head)

    src_tokens = ["Le", "chat", "dort", "sur", "lit"]
    tgt_tokens = ["The", "cat", "sleeps", "on"]

    print("Cross-attention weights (decoder -> encoder):")
    print(f"{'':>10}", "  ".join(f"{t:>6}" for t in src_tokens))
    for i, tgt in enumerate(tgt_tokens):
        row = "  ".join(f"{weights[i,j]:.3f}" for j in range(enc_len))
        print(f"{tgt:>10} {row}")
