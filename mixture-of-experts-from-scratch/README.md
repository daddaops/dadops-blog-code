# Mixture of Experts from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/mixture-of-experts-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `router.py` | Gating network + top-k routing with weight renormalization |
| `load_balance.py` | Auxiliary load-balancing loss (Switch Transformer style) |
| `moe_layer.py` | Complete MoE layer with SwiGLU experts + parameter comparison |
| `training.py` | Forward-only training comparison: MoE vs dense FFN |
| `shared_experts.py` | Shared expert MoE (DeepSeek-V2 style) |
| `aux_free_router.py` | Auxiliary-loss-free bias-based balancing (DeepSeek-V3 style) |

## Usage

```bash
python router.py           # Router demo + top-k routing
python load_balance.py     # Balanced vs collapsed load balance loss
python moe_layer.py        # MoE vs dense FFN parameter comparison
python training.py         # Training loss comparison over 50 steps
python shared_experts.py   # Shared + routed expert combination
python aux_free_router.py  # Bias-based load balancing demo
```

No external dependencies — pure Python + NumPy.
