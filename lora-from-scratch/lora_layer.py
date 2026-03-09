import numpy as np

class LoRALayer:
    """A linear layer with a frozen base weight and trainable low-rank adapters."""

    def __init__(self, W_frozen, rank=8, alpha=None):
        self.W = W_frozen.copy()  # Frozen — never updated
        d, k = W_frozen.shape
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank  # default: alpha = rank

        # A: small random init (Kaiming-style)
        self.A = np.random.randn(rank, k) * np.sqrt(2.0 / k)
        # B: zero init — so B @ A = 0 at start
        self.B = np.zeros((d, rank))

        # Cache for backpropagation
        self._input = None

    def forward(self, x):
        """x shape: (batch, k) → output shape: (batch, d)"""
        self._input = x
        base_out = x @ self.W.T           # Standard linear: (batch, k) @ (k, d) → (batch, d)
        lora_out = x @ self.A.T @ self.B.T  # LoRA path: x → A → B
        scale = self.alpha / self.rank
        return base_out + scale * lora_out

    def trainable_params(self):
        return self.A.size + self.B.size

    def total_base_params(self):
        return self.W.size

if __name__ == "__main__":
    # Compare parameter counts
    d, k = 4096, 4096
    W = np.random.randn(d, k) * 0.01
    layer = LoRALayer(W, rank=8, alpha=16)

    print(f"Base weight params:     {layer.total_base_params():>12,}")
    print(f"LoRA trainable params:  {layer.trainable_params():>12,}")
    print(f"Reduction:              {layer.total_base_params() / layer.trainable_params():>11.0f}x")
