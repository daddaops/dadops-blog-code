"""Shared Expert MoE (DeepSeek-V2 style).

Combines always-active shared experts with selectively-routed experts,
so universal knowledge is always available while specialized processing
is handled by the router.
"""
import numpy as np
from moe_layer import softmax, silu, SwiGLUExpert, MoELayer

class SharedExpertMoE:
    """MoE with always-active shared expert(s) + routed experts."""

    def __init__(self, d_model, d_ff, num_shared=1, num_routed=8, top_k=2):
        # Shared experts: always active for every token
        self.shared = [SwiGLUExpert(d_model, d_ff) for _ in range(num_shared)]

        # Routed experts: selected by the router
        self.routed_moe = MoELayer(d_model, d_ff, num_experts=num_routed, top_k=top_k)

    def __call__(self, x):
        # Shared experts process ALL tokens
        shared_out = sum(expert(x) for expert in self.shared)

        # Routed experts process tokens selectively
        routed_out = self.routed_moe(x)

        # Combine both contributions
        return shared_out + routed_out


if __name__ == "__main__":
    np.random.seed(42)
    d_model, d_ff = 64, 172

    shared_moe = SharedExpertMoE(d_model, d_ff, num_shared=1, num_routed=8, top_k=2)
    x = np.random.randn(10, d_model)
    out = shared_moe(x)
    print(f"SharedExpertMoE output shape: {out.shape}")
    print(f"Aux loss: {shared_moe.routed_moe.aux_loss:.4f}")
