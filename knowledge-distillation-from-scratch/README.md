# Knowledge Distillation from Scratch

Verified, runnable code from the [Knowledge Distillation from Scratch](https://dadops.dev/blog/knowledge-distillation-from-scratch/) blog post.

## Scripts

- **soft_labels.py** — Softmax with temperature, hard vs soft labels (shared module)
- **temperature.py** — Temperature scaling reveals dark knowledge
- **distillation_loss.py** — KL divergence, cross-entropy, and combined distillation loss
- **t_squared.py** — T² gradient rescaling factor derivation and verification
- **training.py** — Full teacher/student training experiment (3 approaches)
- **fitnets.py** — FitNets feature distillation with learned regressor

## Run

```bash
pip install -r requirements.txt
python soft_labels.py
python temperature.py
python distillation_loss.py
python t_squared.py
python training.py
python fitnets.py
```
