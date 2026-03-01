"""
Code Block 3: Run comparison and diff tools.

From: https://dadops.dev/blog/ml-experiment-tracking/

load_all_runs() collects params and metrics from all run directories.
compare_runs() prints a sorted leaderboard.
diff_runs() shows parameter differences between two runs.

No external dependencies required.
"""

import os
import json


def load_all_runs(base_dir="runs"):
    """Load params and final metrics from every run directory."""
    runs = []
    for name in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, name)
        if not os.path.isdir(run_dir):
            continue
        run = {"name": name}
        params_path = os.path.join(run_dir, "params.json")
        summary_path = os.path.join(run_dir, "summary.json")
        if os.path.exists(params_path):
            with open(params_path) as f:
                run["params"] = json.load(f)
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                run["summary"] = json.load(f)
        runs.append(run)
    return runs


def compare_runs(runs, sort_by="val_accuracy", top_n=10):
    """Print a sorted leaderboard of runs."""
    scored = []
    for r in runs:
        metrics = r.get("summary", {}).get("final_metrics", {})
        score = metrics.get(sort_by, 0)
        params = r.get("params", {})
        scored.append((score, r["name"], params.get("lr"), params.get("batch_size")))
    scored.sort(reverse=True)
    print(f"{'Rank':<6}{'Run':<30}{'Score':<10}{'LR':<10}{'Batch':<8}")
    print("-" * 64)
    for i, (score, name, lr, bs) in enumerate(scored[:top_n]):
        print(f"{i+1:<6}{name:<30}{score:<10.4f}{lr!s:<10}{bs!s:<8}")


def diff_runs(run_a, run_b):
    """Show which hyperparameters differ between two runs."""
    params_a = run_a.get("params", {})
    params_b = run_b.get("params", {})
    all_keys = set(params_a) | set(params_b)
    diffs = []
    for key in sorted(all_keys):
        va, vb = params_a.get(key), params_b.get(key)
        if va != vb:
            diffs.append((key, va, vb))
    if diffs:
        print(f"{'Parameter':<20}{'Run A':<20}{'Run B':<20}")
        print("-" * 60)
        for key, va, vb in diffs:
            print(f"{key:<20}{str(va):<20}{str(vb):<20}")
    else:
        print("Runs have identical parameters.")


if __name__ == "__main__":
    import tempfile
    import shutil
    from experiment_tracker import ExperimentTracker
    import random

    print("=== Run Comparison Demo ===\n")
    base = tempfile.mkdtemp()
    random.seed(42)

    # Create 5 mock runs with different hyperparameters
    configs = [
        {"model": "resnet18", "lr": 0.001, "batch_size": 32, "epochs": 10},
        {"model": "resnet18", "lr": 0.003, "batch_size": 64, "epochs": 20},
        {"model": "resnet34", "lr": 0.01, "batch_size": 32, "epochs": 15},
        {"model": "resnet18", "lr": 0.0001, "batch_size": 128, "epochs": 30},
        {"model": "resnet50", "lr": 0.003, "batch_size": 64, "epochs": 20},
    ]

    for cfg in configs:
        tracker = ExperimentTracker(base_dir=base)
        tracker.log_params(cfg)
        val_acc = 0.8 + random.gauss(0, 0.05)
        tracker.finish({"val_accuracy": round(val_acc, 4), "val_loss": round(1 - val_acc, 4)})

    # Load and compare
    runs = load_all_runs(base)
    print(f"Loaded {len(runs)} runs\n")
    compare_runs(runs, sort_by="val_accuracy", top_n=5)

    # Diff top two
    if len(runs) >= 2:
        print(f"\n--- Diff: {runs[0]['name']} vs {runs[1]['name']} ---")
        diff_runs(runs[0], runs[1])

    shutil.rmtree(base)
