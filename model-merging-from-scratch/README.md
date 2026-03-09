# Model Merging from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/model-merging-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `loss_interpolation.py` | Linear interpolation between specialist models on the loss landscape |
| `task_arithmetic.py` | Task vector addition and negation for multi-task merging |
| `ties_merge.py` | TIES-Merging: Trim, Elect Sign, Disjoint Merge |
| `merge_evaluator.py` | SLERP, DARE, and comparison across all merging methods |

## Usage

```bash
python loss_interpolation.py   # Loss landscape interpolation
python task_arithmetic.py      # Task vector addition and negation
python ties_merge.py           # TIES-Merging demo
python merge_evaluator.py      # Full method comparison table
```

No external dependencies — pure Python + NumPy.
