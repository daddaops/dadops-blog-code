"""Minimal character-level language model using a selective SSM.

Embeds characters, passes them through a Mamba-style SSM layer, and
predicts the next character. Demonstrates that the hidden state is
fixed-size regardless of sequence length.
"""
import numpy as np

# Minimal character-level SSM language model
text = "to be or not to be that is the question " * 5
chars = sorted(set(text))
V = len(chars)  # vocab size
ch2ix = {c: i for i, c in enumerate(chars)}

data = np.array([ch2ix[c] for c in text])

# Model hyperparameters
D_emb, N_st = 16, 8
np.random.seed(42)

# Learnable parameters
W_emb = np.random.randn(V, D_emb) * 0.1          # character embeddings
A_log = -np.abs(np.random.randn(D_emb, N_st))     # state decay (fixed per layer)
W_B = np.random.randn(D_emb, N_st) * 0.05         # B projection
W_C = np.random.randn(D_emb, N_st) * 0.05         # C projection
W_dt = np.random.randn(D_emb, 1) * 0.05           # delta projection
b_dt = np.full(1, -3.0)                            # delta bias
W_out = np.random.randn(D_emb, V) * 0.1           # output head

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def forward_ssm(indices):
    """Forward pass: embed → selective SSM → predict next char."""
    L = len(indices)
    h = np.zeros((D_emb, N_st))  # SSM hidden state
    total_loss = 0.0

    for t in range(L - 1):
        x = W_emb[indices[t]]  # (D_emb,)

        # Selective SSM step
        B_t = x @ W_B                                            # (N_st,)
        C_t = x @ W_C                                            # (N_st,)
        dt = np.log1p(np.exp(x @ W_dt + b_dt)).mean()            # softplus

        A_bar = np.exp(dt * A_log)                                # (D_emb, N_st)
        h = A_bar * h + np.outer(x, dt * B_t)                    # state update
        y = (h * C_t).sum(axis=1)                                 # readout (D_emb,)

        # Next-character prediction
        logits = y @ W_out                                        # (V,)
        probs = softmax(logits)
        total_loss -= np.log(probs[indices[t + 1]] + 1e-8)

    return total_loss / (L - 1)

loss = forward_ssm(data)
print(f"Initial loss: {loss:.3f}  (random baseline: {np.log(V):.3f})")
print(f"Vocab: {V} chars | Embedding: {D_emb}d | State: {D_emb}x{N_st} = {D_emb*N_st}")
print(f"The hidden state h is FIXED SIZE — no KV cache, no growing memory.")
print(f"Generate a million tokens? Still just {D_emb*N_st} state values.")
