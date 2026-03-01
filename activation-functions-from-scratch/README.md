# Activation Functions from Scratch — Verified Code

Runnable code from the DadOps blog post: [Activation Functions from Scratch](https://dadops.dev/blog/activation-functions-from-scratch/)

## Scripts

| Script | Description | API Key Required? |
|--------|-------------|-------------------|
| `xor_demo.py` | Why activations matter — XOR with/without nonlinearity | No |
| `sigmoid_tanh.py` | Sigmoid and tanh functions + gradients | No |
| `gradient_flow.py` | Gradient flow simulation through 10 layers | No |
| `relu_family.py` | ReLU, Leaky ReLU, PReLU, ELU, SELU | No |
| `smooth_activations.py` | GELU, SiLU/Swish, Mish | No |
| `softmax_softplus.py` | Softmax and Softplus | No |
| `mlp_comparison.py` | 4-layer MLP trained with 8 activations on spiral data | No |

## Quick Start

```bash
pip install -r requirements.txt

python xor_demo.py
python sigmoid_tanh.py
python gradient_flow.py
python relu_family.py
python smooth_activations.py
python softmax_softplus.py
python mlp_comparison.py
```

## Notes

- All scripts run without API keys — pure numpy/scipy
- `mlp_comparison.py` uses seeded PRNG (seed=42) for reproducible results
- The walrus operator (`:=`) in `mlp_comparison.py` requires Python 3.8+
