# ML Experiment Tracking — Code from Blog Post

Extracted from: https://daddaops.com/blog/ml-experiment-tracking/

## Scripts

| Script | Code Block | Description |
|--------|-----------|-------------|
| `experiment_tracker.py` | 1 | Minimal experiment tracker — timestamped run dirs with params.json, metrics.csv, git_info.json, summary.json |
| `compare_runs.py` | 3 | Run comparison tools — load_all_runs, compare_runs leaderboard, diff_runs |
| `reproducible_pipeline.py` | 4 | Reproducibility controls — seed_everything, ReproduciblePipeline with config-driven training |
| `experiment_registry.py` | 5 | Experiment grouping and model registry — Experiment class, ModelRegistry with promote/archive lifecycle |
| `hyperparameter_search.py` | 6 | Hyperparameter search strategies — SearchSpace, grid_search, random_search, successive_halving |
| `verify_ml_experiment_tracking.py` | — | Verification test suite comparing scripts against blog claims |

## Dependencies

- Python 3.8+
- `numpy` (for `reproducible_pipeline.py` only)
- All other scripts use stdlib only

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run individual scripts
python experiment_tracker.py
python compare_runs.py
python reproducible_pipeline.py
python experiment_registry.py
python hyperparameter_search.py

# Run verification suite
python verify_ml_experiment_tracking.py
```

Code Block 2 is a usage example (not standalone) — it's covered by the experiment_tracker.py demo.
