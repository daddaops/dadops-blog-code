"""ByT5-style Byte Encoder — byte embedding + downsampling demo.

Code Block 3: Shows how byte-level encoders reduce sequence length via pooling.
"""
import numpy as np


class ByteEncoder:
    """Simplified ByT5-style byte-level encoder with downsampling."""

    def __init__(self, d_model=128, n_heads=4, pool_factor=4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.pool_factor = pool_factor
        # Byte embedding: only 256 entries!
        self.embed = np.random.randn(256, d_model) * 0.02

    def encode(self, text):
        """Encode text as raw bytes, embed, and downsample."""
        byte_ids = list(text.encode('utf-8'))
        seq_len = len(byte_ids)

        # Step 1: Byte embedding lookup
        x = np.array([self.embed[b] for b in byte_ids])  # (seq_len, d_model)
        print(f"Input:  '{text}'")
        print(f"Bytes:  {seq_len} positions, shape {x.shape}")

        # Step 2: Add sinusoidal positional encoding
        pos = np.arange(seq_len)[:, None]
        dim = np.arange(0, self.d_model, 2)[None, :]
        angles = pos / (10000 ** (dim / self.d_model))
        x[:, 0::2] += np.sin(angles)
        x[:, 1::2] += np.cos(angles)

        # Step 3: Mean-pool downsampling (ByT5's key trick)
        k = self.pool_factor
        trimmed = seq_len - (seq_len % k)  # trim to multiple of k
        x_trim = x[:trimmed].reshape(-1, k, self.d_model)
        x_pooled = x_trim.mean(axis=1)  # (seq_len/k, d_model)
        print(f"Pooled: {x_pooled.shape[0]} positions (/{k}), shape {x_pooled.shape}")

        return x_pooled


enc = ByteEncoder(d_model=128, pool_factor=4)
enc.encode("Hello, world!")
print()
enc.encode("你好世界")  # Chinese: "Hello world"
print()
# Embedding table comparison
print(f"Byte embedding:    256 x 128 = {256*128:,} params")
print(f"Subword embedding: 50257 x 128 = {50257*128:,} params")
print(f"Reduction: {50257*128 // (256*128)}x fewer embedding params")
