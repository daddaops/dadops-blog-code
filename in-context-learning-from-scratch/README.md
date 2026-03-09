# In-Context Learning from Scratch

Verified, runnable code from the [In-Context Learning from Scratch](https://dadops.dev/blog/in-context-learning-from-scratch/) blog post.

## Scripts

- **attention_gradient_descent.py** — Attention implements gradient descent on in-context examples
- **induction_head_scoring.py** — Scoring attention heads for induction head patterns
- **phase_transition.py** — Simulated ICL phase transition during training
- **task_vectors.py** — Task vector extraction and PCA visualization

## Run

```bash
pip install -r requirements.txt
python attention_gradient_descent.py
python induction_head_scoring.py
python phase_transition.py
python task_vectors.py
```
