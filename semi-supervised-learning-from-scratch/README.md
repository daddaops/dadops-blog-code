# Semi-Supervised Learning from Scratch

Verified, runnable code from the DadOps blog post:
[Semi-Supervised Learning from Scratch](https://www.dadops.co/blog/semi-supervised-learning-from-scratch/)

## Scripts

- **self_training.py** — Self-training with pseudo-labels (logistic regression on two-moons)
- **label_propagation.py** — Label propagation through an RBF similarity graph
- **pi_model.py** — Pi-Model consistency regularization with a single hidden layer network
- **entropy_minimization.py** — Entropy minimization for semi-supervised learning
- **mixmatch.py** — Simplified MixMatch: augment-average-sharpen-mixup

## Usage

```bash
pip install -r requirements.txt
python3 self_training.py
python3 label_propagation.py
python3 pi_model.py
python3 entropy_minimization.py
python3 mixmatch.py
```
