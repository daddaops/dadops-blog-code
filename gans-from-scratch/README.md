# GANs from Scratch

Verified, runnable code from the [GANs from Scratch](https://dadops.dev/blog/gans-from-scratch/) blog post.

## Scripts

- **generator.py** — Generator class (2-layer MLP with backprop)
- **discriminator.py** — Discriminator class (2-layer MLP with sigmoid)
- **losses.py** — GAN loss functions (BCE + non-saturating trick)
- **train_1d_gan.py** — Complete 1D GAN training (learns N(3, 0.5))
- **wgan.py** — Wasserstein Critic + loss function
- **train_wgan_2d.py** — 2D WGAN training on 8-Gaussians

## Run

```bash
pip install -r requirements.txt
python train_1d_gan.py
python train_wgan_2d.py
```
