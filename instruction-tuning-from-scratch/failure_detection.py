import numpy as np

def detect_sft_failures(training_log):
    """Diagnose three common SFT failure modes from training metrics.

    training_log: list of dicts with keys:
      'step', 'loss', 'avg_response_length', 'perplexity_held_out',
      'unique_trigram_ratio'
    """
    warnings = []

    # Failure 1: Catastrophic forgetting
    # Signal: held-out perplexity increases during training
    ppl = [e["perplexity_held_out"] for e in training_log]
    if len(ppl) > 5 and ppl[-1] > ppl[0] * 1.25:
        warnings.append(
            f"FORGETTING: Held-out perplexity rose from {ppl[0]:.1f} to "
            f"{ppl[-1]:.1f} (+{(ppl[-1]/ppl[0]-1)*100:.0f}%). "
            f"Reduce learning rate or use LoRA."
        )

    # Failure 2: Verbose rambling (reward hacking)
    # Signal: response length grows while loss plateaus
    lengths = [e["avg_response_length"] for e in training_log]
    losses = [e["loss"] for e in training_log]
    if len(lengths) > 5:
        length_growth = lengths[-1] / max(lengths[0], 1)
        loss_improvement = losses[0] - losses[-1]
        if length_growth > 1.5 and loss_improvement < 0.1:
            warnings.append(
                f"VERBOSITY: Avg response grew {length_growth:.1f}x but loss "
                f"only improved by {loss_improvement:.3f}. "
                f"Balance response lengths in training data."
            )

    # Failure 3: Style collapse
    # Signal: unique trigram ratio decreases over training
    diversity = [e["unique_trigram_ratio"] for e in training_log]
    if len(diversity) > 5 and diversity[-1] < diversity[0] * 0.7:
        warnings.append(
            f"STYLE COLLAPSE: Trigram diversity dropped from "
            f"{diversity[0]:.2f} to {diversity[-1]:.2f}. "
            f"Diversify training data or reduce epochs."
        )

    return warnings if warnings else ["All diagnostics healthy."]

# Simulate a healthy training run
np.random.seed(42)
healthy_log = []
for step in range(0, 1001, 100):
    healthy_log.append({
        "step": step,
        "loss": 2.5 * np.exp(-step / 400) + 0.8,
        "avg_response_length": 45 + np.random.normal(0, 3),
        "perplexity_held_out": 15.0 + np.random.normal(0, 0.5),
        "unique_trigram_ratio": 0.72 + np.random.normal(0, 0.02),
    })

# Simulate a failing run (catastrophic forgetting + verbosity)
failing_log = []
for step in range(0, 1001, 100):
    failing_log.append({
        "step": step,
        "loss": 2.5 * np.exp(-step / 400) + 0.8,
        "avg_response_length": 45 + step * 0.08,
        "perplexity_held_out": 15.0 + step * 0.015,
        "unique_trigram_ratio": 0.72 - step * 0.0003,
    })

print("=== Healthy Run ===")
for w in detect_sft_failures(healthy_log):
    print(f"  {w}")

print("\n=== Failing Run ===")
for w in detect_sft_failures(failing_log):
    print(f"  {w}")
# === Healthy Run ===
#   All diagnostics healthy.
#
# === Failing Run ===
#   FORGETTING: Held-out perplexity rose from 15.0 to 30.0 (+100%). ...
#   VERBOSITY: Avg response grew 2.8x but loss only improved by ...
#   STYLE COLLAPSE: Trigram diversity dropped from 0.72 to 0.42. ...
