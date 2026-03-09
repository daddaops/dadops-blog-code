"""The logit lens: read a model's predictions at each layer.

Projects the residual stream into vocabulary space at every layer
to watch how the model's prediction evolves from vague to precise.
"""
import numpy as np


def logit_lens_demo():
    """Apply the logit lens to a tiny transformer — read predictions at each layer."""
    np.random.seed(21)
    vocab = ["cat", "dog", "fish", "bird", "tree", "rock", "sky", "sun"]
    n_vocab, d_model, n_layers = len(vocab), 16, 6

    # Random embeddings and layer transformations
    embed = np.random.randn(n_vocab, d_model) * 0.5
    unembed = np.random.randn(d_model, n_vocab) * 0.3
    layers = [np.random.randn(d_model, d_model) * 0.2 for _ in range(n_layers)]

    # Input: token index 0 ("cat")
    x = embed[0]  # (d_model,)
    residual = x.copy()

    print("Logit Lens — predictions at each layer:\n")
    print(f"{'Layer':<8} {'Top-1':<8} {'P(top-1)':<10} {'Top-3 predictions'}")
    print("-" * 55)

    for layer_idx in range(n_layers):
        # Layer transform (simplified: tanh nonlinearity + residual)
        residual = residual + np.tanh(residual @ layers[layer_idx]) * 0.5

        # Logit lens: project to vocab space
        logits = residual @ unembed
        probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()

        # Top-3 predictions
        top3 = np.argsort(probs)[::-1][:3]
        top1_word = vocab[top3[0]]
        top1_prob = probs[top3[0]]
        top3_str = ", ".join(f"{vocab[i]} ({probs[i]:.2f})" for i in top3)
        print(f"L{layer_idx + 1:<6} {top1_word:<8} {top1_prob:<10.3f} {top3_str}")

    print("\nWatch the prediction sharpen — early layers are uncertain,")
    print("later layers converge as the residual stream accumulates information.")


if __name__ == "__main__":
    logit_lens_demo()
