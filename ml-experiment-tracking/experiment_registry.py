"""
Code Block 5: Experiment grouping and model registry.

From: https://dadops.dev/blog/ml-experiment-tracking/

Experiment class groups related runs.
ModelRegistry tracks model versions through development → staging → production.

No external dependencies required.
"""

import os
import json
from datetime import datetime

from experiment_tracker import ExperimentTracker


class Experiment:
    """A named collection of related runs (one hypothesis being tested)."""
    def __init__(self, name, base_dir="experiments"):
        self.name = name
        self.exp_dir = os.path.join(base_dir, name)
        os.makedirs(self.exp_dir, exist_ok=True)

    def new_run(self, tags=None):
        """Create a tracked run within this experiment."""
        tracker = ExperimentTracker(base_dir=self.exp_dir)
        if tags:
            tag_path = os.path.join(tracker.run_dir, "tags.json")
            with open(tag_path, "w") as f:
                json.dump(tags, f, indent=2)
        return tracker


class ModelRegistry:
    """Track model versions and their deployment status."""
    def __init__(self, registry_dir="model_registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.registry_file = os.path.join(registry_dir, "registry.json")
        self.registry = self._load()

    def _load(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file) as f:
                return json.load(f)
        return {"models": []}

    def _save(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register(self, name, run_dir, metrics):
        """Register a model from a completed run."""
        version = len([m for m in self.registry["models"] if m["name"] == name]) + 1
        entry = {
            "name": name, "version": version, "run_dir": run_dir,
            "metrics": metrics, "stage": "development",
            "registered_at": datetime.now().isoformat(),
        }
        self.registry["models"].append(entry)
        self._save()
        return version

    def promote(self, name, version, stage):
        """Move a model version to staging or production."""
        for m in self.registry["models"]:
            # Demote existing production model of the same name
            if m["name"] == name and m["stage"] == stage:
                m["stage"] = "archived"
            if m["name"] == name and m["version"] == version:
                m["stage"] = stage
        self._save()

    def get_production_model(self, name):
        """Get the current production model."""
        for m in reversed(self.registry["models"]):
            if m["name"] == name and m["stage"] == "production":
                return m
        return None


if __name__ == "__main__":
    import tempfile
    import shutil

    print("=== Experiment & Model Registry Demo ===\n")
    base = tempfile.mkdtemp()

    # Create an experiment with runs
    exp = Experiment("lr_sweep_resnet18", base_dir=base)
    run1 = exp.new_run(tags={"hypothesis": "baseline lr=0.001"})
    run1.log_params({"lr": 0.001, "model": "resnet18"})
    run1.finish({"val_accuracy": 0.91})

    run2 = exp.new_run(tags={"hypothesis": "higher lr helps"})
    run2.log_params({"lr": 0.01, "model": "resnet18"})
    run2.finish({"val_accuracy": 0.94})

    print(f"Experiment: {exp.name}")
    print(f"  Run 1: {run1.run_dir}")
    print(f"  Run 2: {run2.run_dir}")

    # Model registry
    reg_dir = os.path.join(base, "registry")
    registry = ModelRegistry(registry_dir=reg_dir)

    v1 = registry.register("resnet18", run1.run_dir, {"accuracy": 0.91})
    v2 = registry.register("resnet18", run2.run_dir, {"accuracy": 0.94})
    print(f"\nRegistered v{v1} (accuracy=0.91)")
    print(f"Registered v{v2} (accuracy=0.94)")

    # Promote v2 to production
    registry.promote("resnet18", version=2, stage="production")
    prod = registry.get_production_model("resnet18")
    print(f"\nProduction model: v{prod['version']} (accuracy={prod['metrics']['accuracy']})")

    # Register v3 and promote — v2 should be archived
    v3 = registry.register("resnet18", run2.run_dir, {"accuracy": 0.95})
    registry.promote("resnet18", version=3, stage="production")

    # Check v2 is now archived
    for m in registry.registry["models"]:
        if m["name"] == "resnet18":
            print(f"  v{m['version']}: stage={m['stage']}")

    shutil.rmtree(base)
