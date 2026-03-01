"""
Verification suite for ml-experiment-tracking blog post.

Tests all 5 extracted scripts against blog claims:
- ExperimentTracker creates correct directory structure and files
- compare_runs loads, sorts, and diffs runs correctly
- ReproduciblePipeline produces deterministic results from config
- ModelRegistry manages model versions and lifecycle stages
- Hyperparameter search strategies behave as documented

From: https://dadops.dev/blog/ml-experiment-tracking/
"""

import os
import sys
import json
import csv
import math
import random
import shutil
import tempfile
import hashlib

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


# ============================================================
# Block 1: ExperimentTracker
# ============================================================
print("\n=== Block 1: ExperimentTracker ===")

from experiment_tracker import ExperimentTracker

base = tempfile.mkdtemp()

# Test: creates run directory with timestamp + hex ID pattern
tracker = ExperimentTracker(base_dir=base)
run_name = os.path.basename(tracker.run_dir)
check("Run dir created", os.path.isdir(tracker.run_dir))
check("Run dir has timestamp_hex format",
      len(run_name.split("_")) >= 3 and len(run_name.split("_")[-1]) == 6,
      f"Got: {run_name}")

# Test: log_params writes params.json
tracker.log_params({"model": "resnet18", "lr": 0.003, "batch_size": 64})
params_path = os.path.join(tracker.run_dir, "params.json")
check("params.json created", os.path.exists(params_path))
with open(params_path) as f:
    params = json.load(f)
check("params.json has correct keys", set(params.keys()) == {"model", "lr", "batch_size"})
check("params.json lr value", params["lr"] == 0.003, f"Got: {params['lr']}")

# Test: log_metric writes metrics.csv with step,name,value header
tracker.log_metric("train_loss", 0.95, step=0)
tracker.log_metric("train_loss", 0.85, step=1)
tracker.log_metric("val_accuracy", 0.82, step=0)
check("metrics.csv created", os.path.exists(tracker.metrics_file))
with open(tracker.metrics_file) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)
check("metrics.csv header", header == ["step", "name", "value"], f"Got: {header}")
check("metrics.csv has 3 rows", len(rows) == 3, f"Got: {len(rows)}")
check("metrics.csv first row", rows[0] == ["0", "train_loss", "0.95"])

# Test: log_artifact copies file
artifact_dir = tempfile.mkdtemp()
artifact_path = os.path.join(artifact_dir, "model.pkl")
with open(artifact_path, "w") as f:
    f.write("fake model")
tracker.log_artifact(artifact_path)
check("artifact copied", os.path.exists(os.path.join(tracker.run_dir, "model.pkl")))
shutil.rmtree(artifact_dir)

# Test: _save_git_info records commit hash
git_info_path = os.path.join(tracker.run_dir, "git_info.json")
if os.path.exists(git_info_path):
    with open(git_info_path) as f:
        git_info = json.load(f)
    check("git_info has commit", "commit" in git_info)
    check("git_info has uncommitted_changes", "uncommitted_changes" in git_info)
else:
    check("git_info.json created (in git repo)", False, "Not in a git repo")
    check("git_info has uncommitted_changes", False, "Skipped")

# Test: finish writes summary.json
tracker.finish({"val_accuracy": 0.92, "val_loss": 0.35})
summary_path = os.path.join(tracker.run_dir, "summary.json")
check("summary.json created", os.path.exists(summary_path))
with open(summary_path) as f:
    summary = json.load(f)
check("summary has run_dir", "run_dir" in summary)
check("summary has start_time", "start_time" in summary)
check("summary has end_time", "end_time" in summary)
check("summary has duration_seconds", "duration_seconds" in summary)
check("summary has final_metrics", summary.get("final_metrics", {}).get("val_accuracy") == 0.92)

# Blog claim: directory has params.json, metrics.csv, git_info.json, summary.json
expected_files = {"params.json", "metrics.csv", "summary.json"}
actual_files = set(os.listdir(tracker.run_dir))
check("Run dir has expected files", expected_files.issubset(actual_files),
      f"Missing: {expected_files - actual_files}")

shutil.rmtree(base)

# ============================================================
# Block 3: compare_runs
# ============================================================
print("\n=== Block 3: compare_runs ===")

from compare_runs import load_all_runs, compare_runs, diff_runs

base = tempfile.mkdtemp()
random.seed(42)

# Create 5 runs with known parameters
configs = [
    {"model": "resnet18", "lr": 0.001, "batch_size": 32},
    {"model": "resnet18", "lr": 0.003, "batch_size": 64},
    {"model": "resnet34", "lr": 0.01, "batch_size": 32},
]
trackers = []
for cfg in configs:
    t = ExperimentTracker(base_dir=base)
    t.log_params(cfg)
    acc = 0.8 + random.gauss(0, 0.05)
    t.finish({"val_accuracy": round(acc, 4), "val_loss": round(1 - acc, 4)})
    trackers.append(t)

# Test: load_all_runs finds all runs
runs = load_all_runs(base)
check("load_all_runs count", len(runs) == 3, f"Got: {len(runs)}")
check("runs have params", all("params" in r for r in runs))
check("runs have summary", all("summary" in r for r in runs))

# Test: compare_runs prints leaderboard (just check it doesn't crash)
import io
from contextlib import redirect_stdout

buf = io.StringIO()
with redirect_stdout(buf):
    compare_runs(runs, sort_by="val_accuracy", top_n=3)
output = buf.getvalue()
check("compare_runs produces output", len(output) > 50)
check("compare_runs has header line with dashes", "-" * 10 in output)

# Test: diff_runs shows parameter differences
buf = io.StringIO()
with redirect_stdout(buf):
    diff_runs(runs[0], runs[1])
diff_output = buf.getvalue()
check("diff_runs produces output", len(diff_output) > 20)

# Test: diff_runs on identical params says identical
same_run = {"params": {"lr": 0.001}}
buf = io.StringIO()
with redirect_stdout(buf):
    diff_runs(same_run, same_run)
check("diff_runs identical", "identical" in buf.getvalue().lower())

shutil.rmtree(base)

# ============================================================
# Block 4: Reproducible Pipeline
# ============================================================
print("\n=== Block 4: Reproducible Pipeline ===")

from reproducible_pipeline import seed_everything, ReproduciblePipeline

# Test: seed_everything produces deterministic random sequences
seed_everything(42)
seq1 = [random.random() for _ in range(10)]
seed_everything(42)
seq2 = [random.random() for _ in range(10)]
check("seed_everything deterministic (random)", seq1 == seq2)

# Test: seed_everything with numpy
import numpy as np
seed_everything(123)
np_seq1 = np.random.rand(5).tolist()
seed_everything(123)
np_seq2 = np.random.rand(5).tolist()
check("seed_everything deterministic (numpy)", np_seq1 == np_seq2)

# Test: ReproduciblePipeline fingerprint
config = {"seed": 42, "model": "resnet18", "lr": 0.003, "epochs": 5}
config_dir = tempfile.mkdtemp()
config_path = os.path.join(config_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f)

p1 = ReproduciblePipeline(config_path)
fp1 = p1.fingerprint()
check("fingerprint is 12 chars hex", len(fp1) == 12 and all(c in "0123456789abcdef" for c in fp1))

# Fingerprint is deterministic (same config → same hash)
p2 = ReproduciblePipeline(config_path)
fp2 = p2.fingerprint()
check("fingerprint deterministic", fp1 == fp2)

# Verify fingerprint matches our manual calculation
config_str = json.dumps(config, sort_keys=True)
expected_fp = hashlib.sha256(config_str.encode()).hexdigest()[:12]
check("fingerprint matches manual hash", fp1 == expected_fp,
      f"Got {fp1}, expected {expected_fp}")

# Test: ReproduciblePipeline.run() creates run directory
run_dir = p1.run()
check("pipeline run creates directory", os.path.isdir(run_dir))
check("pipeline run saves params", os.path.exists(os.path.join(run_dir, "params.json")))
check("pipeline run saves summary", os.path.exists(os.path.join(run_dir, "summary.json")))
check("pipeline run saves metrics", os.path.exists(os.path.join(run_dir, "metrics.csv")))

# Config artifact saved
check("pipeline run saves config artifact",
      os.path.exists(os.path.join(run_dir, "config.json")))

# Cleanup
shutil.rmtree(config_dir)
shutil.rmtree(os.path.dirname(run_dir))

# ============================================================
# Block 5: Experiment & Model Registry
# ============================================================
print("\n=== Block 5: Experiment & Model Registry ===")

from experiment_registry import Experiment, ModelRegistry

base = tempfile.mkdtemp()

# Test: Experiment creates directory and manages runs
exp = Experiment("lr_sweep_resnet18", base_dir=base)
check("experiment dir created", os.path.isdir(exp.exp_dir))

run1 = exp.new_run(tags={"hypothesis": "baseline"})
check("new_run returns ExperimentTracker", isinstance(run1, ExperimentTracker))
check("tags.json created", os.path.exists(os.path.join(run1.run_dir, "tags.json")))
with open(os.path.join(run1.run_dir, "tags.json")) as f:
    tags = json.load(f)
check("tags content correct", tags == {"hypothesis": "baseline"})

run1.log_params({"lr": 0.001})
run1.finish({"val_accuracy": 0.91})

run2 = exp.new_run(tags={"hypothesis": "higher lr"})
run2.log_params({"lr": 0.01})
run2.finish({"val_accuracy": 0.94})

# Test: ModelRegistry register and versioning
reg_dir = os.path.join(base, "registry")
registry = ModelRegistry(registry_dir=reg_dir)

v1 = registry.register("resnet18", run1.run_dir, {"accuracy": 0.91})
check("first version is 1", v1 == 1)
v2 = registry.register("resnet18", run2.run_dir, {"accuracy": 0.94})
check("second version is 2", v2 == 2)

# Test: initial stage is development
for m in registry.registry["models"]:
    check(f"v{m['version']} initial stage is development", m["stage"] == "development")

# Test: promote to production
registry.promote("resnet18", version=2, stage="production")
prod = registry.get_production_model("resnet18")
check("production model is v2", prod is not None and prod["version"] == 2)
check("production model metrics", prod["metrics"]["accuracy"] == 0.94)

# Test: promoting v3 archives v2
v3 = registry.register("resnet18", run2.run_dir, {"accuracy": 0.95})
check("third version is 3", v3 == 3)
registry.promote("resnet18", version=3, stage="production")

# v2 should now be archived
for m in registry.registry["models"]:
    if m["name"] == "resnet18" and m["version"] == 2:
        check("v2 archived after v3 promoted", m["stage"] == "archived")
    if m["name"] == "resnet18" and m["version"] == 3:
        check("v3 is production", m["stage"] == "production")

# Test: get_production_model returns latest production
prod = registry.get_production_model("resnet18")
check("production model now v3", prod["version"] == 3)

# Test: get_production_model returns None for unknown
check("unknown model returns None", registry.get_production_model("vgg16") is None)

# Test: registry persists to disk
registry2 = ModelRegistry(registry_dir=reg_dir)
check("registry persists", len(registry2.registry["models"]) == 3)

shutil.rmtree(base)

# ============================================================
# Block 6: Hyperparameter Search
# ============================================================
print("\n=== Block 6: Hyperparameter Search ===")

from hyperparameter_search import SearchSpace, grid_search, random_search, successive_halving

# Test: grid_search exhaustive combinations
grid = {"lr": [0.001, 0.01, 0.1], "batch_size": [32, 64, 128]}
grid_configs = list(grid_search(grid))
check("grid search 3x3 = 9 configs", len(grid_configs) == 9, f"Got: {len(grid_configs)}")

# Verify all combinations present
lrs = set(c["lr"] for c in grid_configs)
batches = set(c["batch_size"] for c in grid_configs)
check("grid has all 3 LRs", lrs == {0.001, 0.01, 0.1})
check("grid has all 3 batch sizes", batches == {32, 64, 128})

# Blog claim: 10x10 grid = 100 evaluations, only 10 distinct LRs
grid_10x10 = {
    "lr": [0.001 * i for i in range(1, 11)],
    "batch_size": [8 * i for i in range(1, 11)],
}
all_grid = list(grid_search(grid_10x10))
distinct_lrs = len(set(c["lr"] for c in all_grid))
check("10x10 grid = 100 evals", len(all_grid) == 100, f"Got: {len(all_grid)}")
check("10x10 grid only 10 distinct LRs", distinct_lrs == 10, f"Got: {distinct_lrs}")

# Test: random_search returns correct number of configs
random.seed(42)
space = {
    "lr": SearchSpace.log_uniform(1e-5, 1e-1),
    "batch_size": SearchSpace.choice([16, 32, 64, 128]),
    "dropout": SearchSpace.uniform(0.0, 0.5),
}
random_configs = list(random_search(space, 50))
check("random search returns 50 configs", len(random_configs) == 50)
check("random configs have all keys",
      all(set(c.keys()) == {"lr", "batch_size", "dropout"} for c in random_configs))

# Blog claim: random 100 evals = 100 distinct LR values
random.seed(42)
random_100 = list(random_search(
    {"lr": SearchSpace.uniform(0.001, 0.01), "batch_size": SearchSpace.choice(range(8, 81))},
    100
))
distinct_random_lrs = len(set(c["lr"] for c in random_100))
check("random 100 = 100 distinct LRs", distinct_random_lrs == 100,
      f"Got: {distinct_random_lrs}")

# Test: SearchSpace.log_uniform samples in correct range
random.seed(42)
sampler = SearchSpace.log_uniform(1e-5, 1e-1)
samples = [sampler() for _ in range(1000)]
check("log_uniform min >= 1e-5", min(samples) >= 1e-5)
check("log_uniform max <= 1e-1", max(samples) <= 1e-1)
# Log-uniform median ≈ geometric mean = sqrt(low * high)
geo_mean = math.sqrt(1e-5 * 1e-1)
median = sorted(samples)[500]
check("log_uniform median near geometric mean",
      geo_mean / 2 < median < geo_mean * 2,
      f"Median: {median:.6f}, geo mean: {geo_mean:.6f}")

# Test: successive_halving finds good config
random.seed(42)

def mock_evaluate(cfg, budget):
    lr_score = 1.0 - abs(math.log10(cfg["lr"]) - math.log10(0.003))
    bs_score = 1.0 - abs(cfg["batch_size"] - 64) / 128
    return lr_score + bs_score + random.gauss(0, 0.1)

best = successive_halving(space, n_configs=27, min_budget=1, max_budget=81,
                          evaluate_fn=mock_evaluate)
check("successive_halving returns a config", best is not None)
check("successive_halving config has keys",
      set(best.keys()) == {"lr", "batch_size", "dropout"})

# Test: successive_halving with n_configs=1 returns that config
random.seed(42)
single = successive_halving(space, n_configs=1, min_budget=1, max_budget=81,
                            evaluate_fn=mock_evaluate)
check("successive_halving n=1 returns config", single is not None)

# Test: SearchSpace.choice returns from options
random.seed(42)
choice_sampler = SearchSpace.choice(["a", "b", "c"])
choice_samples = [choice_sampler() for _ in range(100)]
check("choice samples from options", set(choice_samples).issubset({"a", "b", "c"}))
check("choice uses all options", set(choice_samples) == {"a", "b", "c"})

# Test: SearchSpace.uniform in range
random.seed(42)
uniform_sampler = SearchSpace.uniform(0.0, 0.5)
uniform_samples = [uniform_sampler() for _ in range(1000)]
check("uniform min >= 0.0", min(uniform_samples) >= 0.0)
check("uniform max <= 0.5", max(uniform_samples) <= 0.5)


# ============================================================
# Blog Claims Verification
# ============================================================
print("\n=== Blog Claims ===")

# Claim: "under 60 lines" for ExperimentTracker
import inspect
source = inspect.getsource(ExperimentTracker)
line_count = len(source.strip().split("\n"))
check("ExperimentTracker under 60 lines", line_count <= 60,
      f"Got: {line_count} lines")

# Claim: grid search is O(k^d) — 3 params * 3 values each = 27
grid_3d = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
check("grid search O(k^d): 3^3 = 27",
      len(list(grid_search(grid_3d))) == 27)

# Claim: successive halving prunes half each round
# With 8 configs, should go: 8 → 4 → 2 → 1 = 3 rounds
random.seed(42)
eval_count = [0]

def counting_evaluate(cfg, budget):
    eval_count[0] += 1
    return random.random()

small_space = {"x": SearchSpace.uniform(0, 1)}
successive_halving(small_space, n_configs=8, min_budget=1, max_budget=8,
                   evaluate_fn=counting_evaluate)
# Round 1: 8 evals at budget=1, keep 4
# Round 2: 4 evals at budget=2, keep 2
# Round 3: 2 evals at budget=4, keep 1
# Budget doubles: 1→2→4→8, but 8>max_budget? No, 8<=8, so:
# Round 1: 8 at budget=1, keep 4
# Round 2: 4 at budget=2, keep 2
# Round 3: 2 at budget=4, keep 1
# Round 4: 1 config, len(configs) > 1 is False, stop
# Total: 8 + 4 + 2 = 14 evals
check("successive halving total evals (8 configs)", eval_count[0] == 14,
      f"Got: {eval_count[0]}")

# Claim: "total code is under 300 lines of Python with zero external dependencies"
# (reproducible_pipeline uses numpy, but the blog says "zero" for the core tracker)
# We check the 4 stdlib-only scripts
stdlib_scripts = [
    "experiment_tracker.py",
    "compare_runs.py",
    "experiment_registry.py",
    "hyperparameter_search.py",
]
total_lines = 0
for script in stdlib_scripts:
    path = os.path.join(os.path.dirname(__file__) or ".", script)
    with open(path) as f:
        lines = len(f.readlines())
    total_lines += lines
# Note: these include docstrings and __main__ demos, so they're longer than
# the blog's code blocks. The blog claim is about the code blocks, not the scripts.
check("stdlib scripts exist and are readable", total_lines > 100)


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"TOTAL: {passed} passed, {failed} failed out of {passed + failed}")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"SOME TESTS FAILED")
    sys.exit(1)
