"""Activation patching on a tiny 3-layer network.

Demonstrates the causal intervention technique: patch clean activations
into a corrupted forward pass to identify which (layer, position) pairs
are causally important for the correct output.
"""
import numpy as np


def activation_patching_demo():
    """Demonstrate activation patching on a tiny 3-layer network."""
    np.random.seed(7)
    n_tokens, d_model, n_layers = 4, 8, 3

    # Simulated model: each layer transforms the residual stream
    # Includes both per-position MLP and cross-position mixing (simulated attention)
    layer_weights = [np.random.randn(d_model, d_model) * 0.3 for _ in range(n_layers)]
    # Position mixing weights simulate attention (causal: each position attends to previous)
    pos_mix = [np.random.randn(n_tokens, n_tokens) * 0.3 for _ in range(n_layers)]
    for pm in pos_mix:
        # Causal mask: position i can only attend to positions <= i
        for i in range(n_tokens):
            pm[i, i+1:] = -1e9
        # Softmax across attended positions
        pm[:] = np.exp(pm - pm.max(axis=-1, keepdims=True))
        pm[:] = pm / pm.sum(axis=-1, keepdims=True)

    unembed = np.random.randn(d_model, 10) * 0.3  # project to 10-class vocab

    def forward(x, patch_layer=None, patch_pos=None, patch_val=None):
        """Forward pass with optional activation patching."""
        residual = x.copy()  # (n_tokens, d_model)
        for layer in range(n_layers):
            # Cross-position mixing (simulated attention)
            mixed = pos_mix[layer] @ residual
            # Per-position MLP
            residual = residual + np.tanh(mixed @ layer_weights[layer])
            if patch_layer == layer and patch_val is not None:
                residual[patch_pos] = patch_val[patch_pos]
        logits = residual @ unembed  # (n_tokens, vocab)
        return residual, logits

    # Clean input: a specific pattern
    x_clean = np.random.randn(n_tokens, d_model)
    clean_residuals = []
    residual = x_clean.copy()
    for layer in range(n_layers):
        mixed = pos_mix[layer] @ residual
        residual = residual + np.tanh(mixed @ layer_weights[layer])
        clean_residuals.append(residual.copy())
    _, clean_logits = forward(x_clean)
    target_class = clean_logits[-1].argmax()  # correct answer at last position

    # Corrupted input: add noise to first two token positions
    x_corrupt = x_clean.copy()
    x_corrupt[:2] += np.random.randn(2, d_model) * 2.0
    _, corrupt_logits = forward(x_corrupt)
    corrupt_prob = np.exp(corrupt_logits[-1]) / np.exp(corrupt_logits[-1]).sum()

    # Patch each (layer, position) and measure recovery
    clean_prob = np.exp(clean_logits[-1]) / np.exp(clean_logits[-1]).sum()
    base_correct = corrupt_prob[target_class]

    print(f"Target class: {target_class}")
    print(f"Clean P(correct):   {clean_prob[target_class]:.3f}")
    print(f"Corrupt P(correct): {base_correct:.3f}\n")
    print("Recovery after patching (layer x position):")
    print(f"{'':^10}", end="")
    for pos in range(n_tokens):
        print(f"Pos {pos:<6}", end="")
    print()

    for layer in range(n_layers):
        print(f"Layer {layer}:  ", end="")
        for pos in range(n_tokens):
            _, patched_logits = forward(
                x_corrupt, patch_layer=layer,
                patch_pos=pos, patch_val=clean_residuals[layer]
            )
            patched_prob = np.exp(patched_logits[-1]) / np.exp(patched_logits[-1]).sum()
            recovery = (patched_prob[target_class] - base_correct) / (clean_prob[target_class] - base_correct + 1e-8)
            print(f"{recovery:<6.2f} ", end="")
        print()


if __name__ == "__main__":
    activation_patching_demo()
