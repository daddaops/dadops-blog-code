"""
Bahdanau Attention + Seq2Seq

Adds attention to the LSTM seq2seq model, solving the bottleneck problem.
Instead of compressing the entire input into one vector, the decoder
can attend to ALL encoder hidden states at each step.

Blog post: https://dadops.dev/blog/encoder-decoder-from-scratch/
"""
import numpy as np
from lstm_seq2seq import sigmoid, tanh_act, lstm_cell


def attention(dec_hidden, enc_outputs, W1, W2, v):
    """Bahdanau additive attention."""
    scores = np.array([
        v @ tanh_act(W1 @ enc_h + W2 @ dec_hidden)
        for enc_h in enc_outputs
    ])
    weights = np.exp(scores - scores.max())
    weights = weights / weights.sum()
    context = weights @ enc_outputs
    return context, weights


def encode_all(sequence, emb, W_enc, b_enc, hidden_dim):
    """Encode and return ALL hidden states (not just the last one)."""
    h = np.zeros(hidden_dim)
    c = np.zeros(hidden_dim)
    all_h = []
    for token in sequence:
        x = emb[token]
        h, c = lstm_cell(x, h, c, W_enc, b_enc)
        all_h.append(h.copy())
    return np.array(all_h), h, c


def decode_with_attention(enc_outputs, init_h, init_c, target_len,
                          emb, W_dec, b_dec, W_out, b_out,
                          W_a1, W_a2, v_a, vocab_size):
    """Decode with attention over encoder outputs."""
    h, c = init_h, init_c
    outputs, all_weights = [], []
    token = 0
    for _ in range(target_len):
        ctx, weights = attention(h, enc_outputs, W_a1, W_a2, v_a)
        all_weights.append(weights)
        x = np.concatenate([emb[token], ctx])
        h, c = lstm_cell(x, h, c, W_dec, b_dec)
        logits = W_out @ h + b_out
        token = np.argmax(logits)
        outputs.append(token)
    return outputs, np.array(all_weights)


if __name__ == "__main__":
    np.random.seed(42)
    vocab_size, emb_dim, hidden_dim = 6, 8, 32
    emb = np.random.randn(vocab_size, emb_dim) * 0.1
    W_enc = np.random.randn(4*hidden_dim, emb_dim+hidden_dim) * 0.05
    b_enc = np.zeros(4*hidden_dim)

    # Setup attention parameters
    attn_dim = 16
    W_a1 = np.random.randn(attn_dim, hidden_dim) * 0.05
    W_a2 = np.random.randn(attn_dim, hidden_dim) * 0.05
    v_a = np.random.randn(attn_dim) * 0.05

    # Decoder takes emb_dim + hidden_dim (token + attention context)
    W_dec_attn = np.random.randn(4*hidden_dim, emb_dim+hidden_dim+hidden_dim) * 0.05
    b_dec = np.zeros(4*hidden_dim)
    W_out = np.random.randn(vocab_size, hidden_dim) * 0.05
    b_out = np.zeros(vocab_size)

    # Encode with all states preserved
    enc_states, h_final, c_final = encode_all([1, 2, 3, 4], emb, W_enc, b_enc, hidden_dim)

    print("Attention-Augmented Seq2Seq")
    print("=" * 50)
    print(f"Encoder outputs shape: {enc_states.shape}")
    print(f"Bottleneck model: 1 vector of dim {hidden_dim}")
    print(f"Attention model:  {enc_states.shape[0]} vectors of dim {hidden_dim} — no information lost")
