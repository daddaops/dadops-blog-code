# Federated Learning from Scratch

Verified, runnable code from the [Federated Learning from Scratch](https://dadops.dev/blog/federated-learning-from-scratch/) blog post.

## Scripts

- **setup.py** — Shared data generation (5 hospitals, 40 patients each)
- **local_vs_pooled.py** — Local vs centralized training comparison
- **fedavg.py** — Federated Averaging (McMahan et al. 2017)
- **sparse_fedavg.py** — Top-k gradient sparsification with error feedback
- **fedprox.py** — FedProx for non-IID data distributions
- **secure_agg.py** — Secure aggregation via pairwise masking
- **dp_fedavg.py** — Differential privacy with clipping and Gaussian noise

## Run

```bash
pip install -r requirements.txt
python local_vs_pooled.py
python fedavg.py
python sparse_fedavg.py
python fedprox.py
python secure_agg.py
python dp_fedavg.py
```
