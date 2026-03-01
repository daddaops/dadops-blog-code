"""
Code Block 1: Minimal experiment tracker from scratch.

From: https://dadops.dev/blog/ml-experiment-tracking/

Under 60 lines, stdlib-only. Creates timestamped run directories
with params.json, metrics.csv, git_info.json, and summary.json.

No external dependencies required.
"""

import os
import json
import csv
import shutil
import subprocess
from datetime import datetime


class ExperimentTracker:
    def __init__(self, base_dir="runs"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        short_id = os.urandom(3).hex()
        self.run_dir = os.path.join(base_dir, f"{timestamp}_{short_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.run_dir, "metrics.csv")
        self.start_time = datetime.now()
        self._save_git_info()

    def log_params(self, params):
        """Save hyperparameters as human-readable JSON."""
        path = os.path.join(self.run_dir, "params.json")
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    def log_metric(self, name, value, step=None):
        """Append a metric to the CSV log (supports time-series)."""
        file_exists = os.path.exists(self.metrics_file)
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "name", "value"])
            writer.writerow([step, name, value])

    def log_artifact(self, filepath):
        """Copy a file into the run directory."""
        dest = os.path.join(self.run_dir, os.path.basename(filepath))
        shutil.copy2(filepath, dest)

    def _save_git_info(self):
        """Record the exact code state."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            diff = subprocess.check_output(
                ["git", "diff", "--stat"], text=True
            ).strip()
            info = {"commit": commit, "uncommitted_changes": diff or "none"}
            path = os.path.join(self.run_dir, "git_info.json")
            with open(path, "w") as f:
                json.dump(info, f, indent=2)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # not in a git repo — skip

    def finish(self, final_metrics=None):
        """Write a summary with runtime and final metrics."""
        summary = {
            "run_dir": self.run_dir,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "final_metrics": final_metrics or {},
        }
        path = os.path.join(self.run_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Run saved to {self.run_dir}")


if __name__ == "__main__":
    import tempfile

    print("=== ExperimentTracker Demo ===\n")
    base = tempfile.mkdtemp()

    tracker = ExperimentTracker(base_dir=base)
    tracker.log_params({
        "model": "resnet18",
        "lr": 0.003,
        "batch_size": 64,
        "epochs": 20,
        "dropout": 0.3,
        "optimizer": "adam",
        "data_version": "v2.1",
    })

    # Simulate 5 epochs of training
    import random
    random.seed(42)
    for epoch in range(5):
        train_loss = 1.0 - epoch * 0.15 + random.gauss(0, 0.02)
        val_loss = 1.1 - epoch * 0.12 + random.gauss(0, 0.03)
        val_acc = 0.5 + epoch * 0.08 + random.gauss(0, 0.01)
        tracker.log_metric("train_loss", round(train_loss, 4), step=epoch)
        tracker.log_metric("val_loss", round(val_loss, 4), step=epoch)
        tracker.log_metric("val_accuracy", round(val_acc, 4), step=epoch)

    tracker.finish({"val_accuracy": round(val_acc, 4), "val_loss": round(val_loss, 4)})

    # Show what was created
    print(f"\nRun directory contents:")
    for f in sorted(os.listdir(tracker.run_dir)):
        path = os.path.join(tracker.run_dir, f)
        print(f"  {f} ({os.path.getsize(path)} bytes)")

    # Show params
    with open(os.path.join(tracker.run_dir, "params.json")) as f:
        print(f"\nparams.json:\n{f.read()}")

    # Show metrics
    with open(os.path.join(tracker.run_dir, "metrics.csv")) as f:
        print(f"metrics.csv:\n{f.read()}")

    # Cleanup
    shutil.rmtree(base)
