"""
FFN as Key-Value Memory

Reinterprets FFN neurons as key-value memory slots where W1 columns
are keys (pattern detectors) and W2 rows are values (stored predictions).

Blog post: https://dadops.dev/blog/ffn-from-scratch/
"""
import numpy as np

np.random.seed(42)


def ffn_as_memory(x, W1, W2, top_k=5):
    """Decompose FFN into individual neuron (memory slot) contributions."""
    d_ff = W1.shape[1]

    scores = x @ W1  # (d_model,) @ (d_model, d_ff) = (d_ff,)
    activations = np.maximum(0, scores)  # ReLU gate

    # Find the top-k most activated memory slots
    top_indices = np.argsort(activations)[-top_k:][::-1]

    print("Top activated memory slots:")
    print(f"{'Slot':>6} | {'Score':>8} | {'Activated':>9}")
    print("-" * 35)
    for idx in top_indices:
        print(f"  {idx:4d} | {scores[idx]:8.3f} | {activations[idx]:9.3f}")

    # The full output is the sum of activated value vectors
    output = activations @ W2  # equivalent to sum_i(act_i * v_i)

    # Show that individual contributions add up to the full FFN output
    manual_sum = np.zeros_like(x)
    for i in range(d_ff):
        manual_sum += activations[i] * W2[i, :]

    print(f"\nDirect FFN output:       {output[:4].round(4)}")
    print(f"Sum of memory values:    {manual_sum[:4].round(4)}")
    print(f"Match: {np.allclose(output, manual_sum)}")

    return output, top_indices


# Demo with a small FFN
d_model, d_ff = 8, 32
W1 = np.random.randn(d_model, d_ff) * 0.5
W2 = np.random.randn(d_ff, d_model) * 0.5
x = np.random.randn(d_model)

output, top_slots = ffn_as_memory(x, W1, W2, top_k=5)
