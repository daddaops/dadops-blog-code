# Mechanistic Interpretability from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/mechanistic-interpretability-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `superposition.py` | Toy model of superposition — compresses sparse features into a bottleneck |
| `probing.py` | Linear probes for concept detection at each layer of an MLP |
| `activation_patching.py` | Activation patching to find causally important components |
| `logit_lens.py` | The logit lens — reading predictions at each layer |
| `attention_heads.py` | Attention head taxonomy — classifying head types |
| `sae_features.py` | Sparse autoencoder for disentangling polysemantic activations |

## Usage

```bash
pip install -r requirements.txt
python superposition.py
python probing.py
python activation_patching.py
python logit_lens.py
python attention_heads.py
python sae_features.py
```
