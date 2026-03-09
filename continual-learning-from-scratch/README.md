# Continual Learning from Scratch — Code Blocks

Extracted from the [Continual Learning from Scratch](https://dadops.dev/blog/continual-learning-from-scratch/) blog post.

## Scripts

| # | Script | Description |
|---|--------|-------------|
| 1 | `01_catastrophic_forgetting.py` | Demonstrates catastrophic forgetting: train on two sequential tasks, watch task 1 accuracy collapse to ~50% |
| 2 | `02_ewc.py` | Elastic Weight Consolidation (EWC) using diagonal Fisher Information to protect important weights |
| 3 | `03_experience_replay.py` | Experience replay with reservoir sampling — mix old buffered examples with new task data |
| 4 | `04_packnet.py` | PackNet — prune weights after each task and freeze them, train freed capacity on the next task |
| 5 | `05_lwf.py` | Learning without Forgetting (LwF) — self-distillation from a frozen snapshot of the old model |
| 6 | `06_metrics.py` | Continual learning metrics: T×T accuracy matrix, average accuracy, and forgetting measure across 5 tasks |

## Shared Module

- `shared.py` — Common utilities: `make_task`, `sigmoid`, `train_mlp`, `accuracy`, `init_weights`

## Requirements

```
numpy
```

## Running

```bash
pip install -r requirements.txt
python 01_catastrophic_forgetting.py
python 02_ewc.py
python 03_experience_replay.py
python 04_packnet.py
python 05_lwf.py
python 06_metrics.py
```
