# Neural Processes from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/neural-processes-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `cnp.py` | Conditional Neural Process with encoder/decoder |
| `permutation_test.py` | Verify permutation invariance of mean pooling |
| `latent_np.py` | Latent NP with reparameterization trick |
| `attentive_np.py` | Attentive NP with cross-attention |
| `train_cnp.py` | Episodic meta-learning training loop |
| `gp_comparison.py` | GP vs NP inference speed and uncertainty |

## Usage

```bash
python cnp.py              # CNP forward pass demo
python permutation_test.py # Permutation invariance check
python latent_np.py        # Latent NP function sampling
python attentive_np.py     # Attentive NP with attention
python train_cnp.py        # Meta-learning training loop
python gp_comparison.py    # GP vs NP comparison
```

Dependencies: numpy.
