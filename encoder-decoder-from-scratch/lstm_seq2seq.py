"""
LSTM Seq2Seq Encoder-Decoder

Implements the original sequence-to-sequence model (Sutskever et al. 2014):
- LSTM encoder reads input and produces a context vector
- LSTM decoder generates output from the context vector
- Demonstrates the bottleneck problem: one vector for the whole sequence

Blog post: https://dadops.dev/blog/encoder-decoder-from-scratch/
"""
import numpy as np


def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
def tanh_act(x): return np.tanh(np.clip(x, -15, 15))


def lstm_cell(x, h, c, W, b):
    """Single LSTM step: x (input_dim,), h (hidden,), c (hidden,)"""
    concat = np.concatenate([x, h])
    gates = W @ concat + b
    hid = h.shape[0]
    f = sigmoid(gates[:hid])
    i = sigmoid(gates[hid:2*hid])
    o = sigmoid(gates[2*hid:3*hid])
    g = tanh_act(gates[3*hid:])
    c_new = f * c + i * g
    h_new = o * tanh_act(c_new)
    return h_new, c_new


def encode(sequence, emb, W_enc, b_enc, hidden_dim):
    """Read input sequence, return final hidden state."""
    h = np.zeros(hidden_dim)
    c = np.zeros(hidden_dim)
    for token in sequence:
        x = emb[token]
        h, c = lstm_cell(x, h, c, W_enc, b_enc)
    return h, c


def decode(context_h, context_c, target_len, emb, W_dec, b_dec, W_out, b_out, vocab_size):
    """Generate output from context vector using greedy decoding."""
    h, c = context_h, context_c
    outputs = []
    token = 0  # Start token
    for _ in range(target_len):
        x = emb[token]
        h, c = lstm_cell(x, h, c, W_dec, b_dec)
        logits = W_out @ h + b_out
        token = np.argmax(logits)
        outputs.append(token)
    return outputs


if __name__ == "__main__":
    # Setup: vocab = {0: START, 1: A, 2: B, 3: C, 4: D, 5: E}
    np.random.seed(42)
    vocab_size, emb_dim, hidden_dim = 6, 8, 32
    emb = np.random.randn(vocab_size, emb_dim) * 0.1
    W_enc = np.random.randn(4*hidden_dim, emb_dim+hidden_dim) * 0.05
    b_enc = np.zeros(4*hidden_dim)
    W_dec = np.random.randn(4*hidden_dim, emb_dim+hidden_dim) * 0.05
    b_dec = np.zeros(4*hidden_dim)
    W_out = np.random.randn(vocab_size, hidden_dim) * 0.05
    b_out = np.zeros(vocab_size)

    # Test reversal: input [1,2,3,4] (ABCD) -> should output [4,3,2,1] (DCBA)
    inp = [1, 2, 3, 4]
    h_ctx, c_ctx = encode(inp, emb, W_enc, b_enc, hidden_dim)
    result = decode(h_ctx, c_ctx, 4, emb, W_dec, b_dec, W_out, b_out, vocab_size)

    print("LSTM Seq2Seq (Bottleneck Model)")
    print("=" * 50)
    print(f"Context vector shape: {h_ctx.shape}")
    print(f"Input:  {inp}")
    print(f"Output: {result}")
    print("(Random before training — architecture is correct)")
