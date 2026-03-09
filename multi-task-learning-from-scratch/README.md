# Multi-Task Learning from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/multi-task-learning-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `multi_task_net.py` | Shared encoder with separate regression/classification heads |
| `uniform_weighting.py` | Demonstrates how uniform weighting lets regression dominate |
| `uncertainty_weighting.py` | Kendall et al. 2018 automatic task weight learning |
| `pcgrad.py` | PCGrad conflict resolution between task gradients |
| `gradnorm.py` | GradNorm: balance training rates via gradient norm matching |
| `train_mtl.py` | Full training loop combining uncertainty weighting + PCGrad |

## Usage

```bash
python multi_task_net.py         # Shared network architecture demo
python uniform_weighting.py      # Uniform weighting problem demo
python uncertainty_weighting.py  # Automatic weight learning
python pcgrad.py                 # Gradient conflict resolution
python gradnorm.py               # Gradient norm balancing
python train_mtl.py              # Full MTL training loop
```

Dependencies: numpy.
