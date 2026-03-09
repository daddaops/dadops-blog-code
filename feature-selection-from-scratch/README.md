# Feature Selection from Scratch

Verified, runnable code from the [Feature Selection from Scratch](https://dadops.dev/blog/feature-selection-from-scratch/) blog post.

## Scripts

- **dataset.py** — Shared dataset generation (3 informative, 3 redundant, 4 noise features)
- **filter_methods.py** — Variance threshold, Pearson correlation, mutual information
- **mrmr_selection.py** — Minimum Redundancy Maximum Relevance greedy selection
- **wrapper_methods.py** — Forward selection and Recursive Feature Elimination
- **lasso_path.py** — Lasso regularization path via coordinate descent
- **permutation_importance.py** — Model-agnostic permutation importance
- **stability_selection.py** — Bootstrap Lasso with selection frequency tracking

## Run

```bash
pip install -r requirements.txt
python filter_methods.py
python mrmr_selection.py
python wrapper_methods.py
python lasso_path.py
python permutation_importance.py
python stability_selection.py
```
