# Positional Encoding from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/positional-encoding-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `attention_permutation.py` | Shows attention is permutation-invariant |
| `sinusoidal_encoding.py` | Vaswani et al. sinusoidal PE |
| `sinusoidal_rotation_verify.py` | Verifies PE(pos+k) = M_k @ PE(pos) |
| `learned_position_embeddings.py` | GPT-2/BERT learned position lookup |
| `rope_implementation.py` | Rotary Position Embeddings (LLaMA/Mistral) |
| `rope_relative_position_proof.py` | Proves RoPE encodes relative position |

## Usage

```bash
python attention_permutation.py        # Permutation invariance demo
python sinusoidal_encoding.py          # Sin/cos PE vectors
python sinusoidal_rotation_verify.py   # Linear transformability proof
python learned_position_embeddings.py  # Learned embeddings
python rope_implementation.py          # RoPE rotations
python rope_relative_position_proof.py # Relative position invariance
```

Dependencies: numpy.
