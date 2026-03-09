"""
Transformer Encoder-Decoder

Implements the full transformer encoder-decoder architecture (Vaswani et al. 2017):
- Multi-head self-attention + FFN encoder layers
- Masked self-attention + cross-attention + FFN decoder layers
- Demonstrates bidirectional encoder vs causal decoder

Blog post: https://dadops.dev/blog/encoder-decoder-from-scratch/
"""
import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def multihead_attention(Q, K, V, W_q, W_k, W_v, W_o, n_heads, mask=None):
    """Multi-head attention. Q,K,V: (seq_len, d_model)"""
    d_model = Q.shape[-1]
    d_head = d_model // n_heads
    q = (Q @ W_q).reshape(-1, n_heads, d_head).transpose(1, 0, 2)
    k = (K @ W_k).reshape(-1, n_heads, d_head).transpose(1, 0, 2)
    v = (V @ W_v).reshape(-1, n_heads, d_head).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_head)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    out = (weights @ v).transpose(1, 0, 2).reshape(-1, d_model)
    return out @ W_o, weights[0]


def ffn(x, W1, b1, W2, b2):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2


def encoder_layer(x, params):
    """One encoder layer: self-attention + FFN with residual + layer norm."""
    attn_out, _ = multihead_attention(x, x, x,
        params['Wq_s'], params['Wk_s'], params['Wv_s'], params['Wo_s'],
        n_heads=2)
    x = layer_norm(x + attn_out)
    ff_out = ffn(x, params['W1'], params['b1'], params['W2'], params['b2'])
    return layer_norm(x + ff_out)


def decoder_layer(x, enc_out, params, causal_mask):
    """One decoder layer: masked self-attn + cross-attn + FFN."""
    self_attn, _ = multihead_attention(x, x, x,
        params['Wq_s'], params['Wk_s'], params['Wv_s'], params['Wo_s'],
        n_heads=2, mask=causal_mask)
    x = layer_norm(x + self_attn)
    cross_attn, cross_weights = multihead_attention(x, enc_out, enc_out,
        params['Wq_c'], params['Wk_c'], params['Wv_c'], params['Wo_c'],
        n_heads=2)
    x = layer_norm(x + cross_attn)
    ff_out = ffn(x, params['W1'], params['b1'], params['W2'], params['b2'])
    return layer_norm(x + ff_out), cross_weights


def make_causal_mask(seq_len):
    mask = np.full((seq_len, seq_len), -1e9)
    return np.triu(mask, k=1)


if __name__ == "__main__":
    np.random.seed(42)
    vocab_size = 6
    d_model, n_heads = 32, 2
    d_ff = 64

    def init_attn_params():
        return {
            'Wq_s': np.random.randn(d_model, d_model) * 0.05,
            'Wk_s': np.random.randn(d_model, d_model) * 0.05,
            'Wv_s': np.random.randn(d_model, d_model) * 0.05,
            'Wo_s': np.random.randn(d_model, d_model) * 0.05,
            'W1': np.random.randn(d_model, d_ff) * 0.05, 'b1': np.zeros(d_ff),
            'W2': np.random.randn(d_ff, d_model) * 0.05, 'b2': np.zeros(d_model),
        }

    def init_decoder_params():
        p = init_attn_params()
        p.update({
            'Wq_c': np.random.randn(d_model, d_model) * 0.05,
            'Wk_c': np.random.randn(d_model, d_model) * 0.05,
            'Wv_c': np.random.randn(d_model, d_model) * 0.05,
            'Wo_c': np.random.randn(d_model, d_model) * 0.05,
        })
        return p

    enc_params = init_attn_params()
    dec_params = init_decoder_params()
    emb_table = np.random.randn(vocab_size, d_model) * 0.1

    # Encode: input tokens -> bidirectional self-attention (no mask!)
    enc_input = emb_table[[1, 2, 3, 4]]
    enc_output = encoder_layer(enc_input, enc_params)

    # Decode: output tokens -> masked self-attention + cross-attention
    dec_input = emb_table[[0, 4, 3, 2]]  # START, D, C, B (teacher forcing)
    mask = make_causal_mask(4)
    dec_output, cross_wts = decoder_layer(dec_input, enc_output, dec_params, mask)

    print("Transformer Encoder-Decoder")
    print("=" * 55)
    print(f"Encoder output: {enc_output.shape}")
    print(f"Decoder output: {dec_output.shape}")
    print(f"Cross-attention weights shape: {cross_wts.shape}")
    print(f"\nKey difference from decoder-only:")
    print(f"  Encoder sees ALL positions (bidirectional)")
    print(f"  Decoder sees past positions (causal) + ALL encoder positions (cross-attn)")
