# Micrograd from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/micrograd-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `micrograd.py` | Complete autograd engine: Value class with forward/backward ops |
| `train.py` | Training a small MLP on a toy dataset using the autograd engine |

## Usage

```bash
python micrograd.py   # Demo: single neuron forward + backward
python train.py       # Train MLP on 4 examples
```

No dependencies needed — pure Python + math stdlib.
