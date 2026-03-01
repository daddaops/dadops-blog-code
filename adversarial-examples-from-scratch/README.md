# Adversarial Examples from Scratch

Verified, runnable code from the [Adversarial Examples from Scratch](https://dadops.dev/blog/adversarial-examples-from-scratch/) blog post.

## Scripts

| Script | Description |
|--------|-------------|
| `fgsm_binary.py` | FGSM attack on a binary MLP classifier (2D ring data) |
| `linearity_hypothesis.py` | Why adversarial attacks work: output shift scales with dimension |
| `fgsm_multiclass.py` | FGSM attack success rate vs epsilon on 3-class problem |
| `pgd_attack.py` | PGD (iterated FGSM) vs single-step FGSM comparison |
| `adversarial_training.py` | Standard vs adversarially-trained model robustness comparison |

## Quick Start

```bash
pip install -r requirements.txt
python fgsm_binary.py
python linearity_hypothesis.py
python fgsm_multiclass.py
python pgd_attack.py
python adversarial_training.py
```

## Dependencies

- numpy (only dependency — all networks are implemented from scratch)
