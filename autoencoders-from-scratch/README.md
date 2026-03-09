# Autoencoders & VAEs from Scratch

Verified, runnable code from the [Autoencoders & VAEs from Scratch](https://www.dadops.co/blog/autoencoders-from-scratch/) blog post.

## Scripts

- `vanilla_autoencoder.py` — Vanilla autoencoder (64→32→2→32→64) trained on 8×8 digits with manual backprop
- `vae.py` — Variational Autoencoder with reparameterization trick, BCE + KL loss
- `beta_vae.py` — β-VAE comparison showing effect of β on reconstruction vs regularization

## Run

```bash
pip install -r requirements.txt
python3 vanilla_autoencoder.py
python3 vae.py
python3 beta_vae.py
```
