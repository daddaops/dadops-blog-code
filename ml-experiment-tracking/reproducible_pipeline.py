"""
Code Block 4: Reproducibility controls — seeding and config-driven pipelines.

From: https://dadops.dev/blog/ml-experiment-tracking/

seed_everything() fixes all randomness sources.
ReproduciblePipeline runs a deterministic pipeline from a JSON config.

Dependencies: numpy (for np.random.seed)
"""

import os
import json
import random
import hashlib
import numpy as np

from experiment_tracker import ExperimentTracker


def seed_everything(seed=42):
    """Fix all sources of randomness for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For PyTorch (if available):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class ReproduciblePipeline:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = json.load(f)
        seed_everything(self.config.get("seed", 42))

    def run(self):
        """Every step is determined by the config — nothing is implicit."""
        cfg = self.config
        tracker = ExperimentTracker()
        tracker.log_params(cfg)
        tracker.log_artifact(self.config_path)  # save the config itself

        # Data loading — version controlled
        # X_train, y_train = load_data(cfg["data_path"], cfg["data_version"])

        # Preprocessing — parameterized, not hardcoded
        # if cfg.get("normalize", True):
        #     mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        #     X_train = (X_train - mean) / std

        # Model — architecture from config
        # model = build_model(cfg["model"], cfg["hidden_size"], cfg["dropout"])

        # Training loop (simulated)
        loss = 1.0
        for epoch in range(cfg.get("epochs", 10)):
            loss = loss * 0.9 + random.gauss(0, 0.01)
            tracker.log_metric("train_loss", round(loss, 4), step=epoch)

        tracker.finish({"final_loss": round(loss, 4)})
        return tracker.run_dir

    def fingerprint(self):
        """Hash the config to uniquely identify this experiment."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


if __name__ == "__main__":
    import tempfile
    import shutil

    print("=== Reproducible Pipeline Demo ===\n")

    # Create a config file
    config = {
        "seed": 42,
        "model": "resnet18",
        "lr": 0.003,
        "batch_size": 64,
        "epochs": 5,
        "dropout": 0.3,
    }
    config_path = os.path.join(tempfile.mkdtemp(), "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Run the pipeline twice and verify fingerprints match
    p1 = ReproduciblePipeline(config_path)
    fp1 = p1.fingerprint()
    run1 = p1.run()

    p2 = ReproduciblePipeline(config_path)
    fp2 = p2.fingerprint()
    run2 = p2.run()

    print(f"\nFingerprint 1: {fp1}")
    print(f"Fingerprint 2: {fp2}")
    print(f"Fingerprints match: {fp1 == fp2}")

    # Verify seeds produce same random sequence
    seed_everything(42)
    seq1 = [random.random() for _ in range(5)]
    seed_everything(42)
    seq2 = [random.random() for _ in range(5)]
    print(f"\nRandom sequences match: {seq1 == seq2}")
    print(f"  Sequence: {[round(x, 6) for x in seq1]}")

    # Cleanup — both runs share the same parent "runs/" dir
    shutil.rmtree(os.path.dirname(config_path))
    runs_dir = os.path.dirname(run1)
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
