# Implicit Bias of Gradient Descent from Scratch

Verified, runnable code from the [Implicit Bias of Gradient Descent from Scratch](https://dadops.dev/blog/implicit-bias-of-gradient-descent-from-scratch/) blog post.

## Scripts

- **minimum_norm.py** — GD finds minimum-norm interpolating solution
- **gd_vs_pseudoinverse.py** — GD converges to pseudoinverse (minimum-norm) solution
- **logistic_max_margin.py** — Logistic regression GD converges to max-margin direction
- **matrix_completion.py** — Deep factorization induces implicit low-rank bias
- **edge_of_stability.py** — Sharpness converges to 2/lr (edge of stability)
- **sgd_batch_size.py** — Smaller batch sizes improve generalization

## Run

```bash
pip install -r requirements.txt
python minimum_norm.py
python gd_vs_pseudoinverse.py
python logistic_max_margin.py
python matrix_completion.py
python edge_of_stability.py
python sgd_batch_size.py
```
