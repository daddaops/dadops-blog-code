# Energy-Based Models from Scratch

Verified, runnable code from the [Energy-Based Models from Scratch](https://dadops.dev/blog/energy-based-models-from-scratch/) blog post.

## Scripts

- **energy_landscape.py** — 2D energy landscape with Gaussian wells and partition function approximation
- **hopfield_network.py** — Hopfield network with Hebbian learning and pattern recall
- **rbm_cd.py** — Restricted Boltzmann Machine trained with Contrastive Divergence (CD-1)
- **langevin_dynamics.py** — MCMC sampling from energy landscape using Langevin dynamics
- **score_matching.py** — Denoising score matching loss function
- **infonce_loss.py** — InfoNCE contrastive loss as energy-based softmax

## Run

```bash
pip install -r requirements.txt
python energy_landscape.py
python hopfield_network.py
python rbm_cd.py
python langevin_dynamics.py
python score_matching.py
python infonce_loss.py
```
