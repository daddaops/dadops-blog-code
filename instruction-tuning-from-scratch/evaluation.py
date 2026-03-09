import numpy as np

def evaluate_instruction_following(responses, references, mode="sft"):
    """Evaluate model responses on 5 dimensions.
    Returns scores per dimension (0.0 to 1.0)."""
    np.random.seed(42 if mode == "base" else 99)
    n = len(responses)

    # Simulated scoring (in practice, use LLM-as-judge or human eval)
    if mode == "base":
        format_scores = np.clip(np.random.normal(0.15, 0.10, n), 0, 1)
        adherence     = np.clip(np.random.normal(0.20, 0.12, n), 0, 1)
        conciseness   = np.clip(np.random.normal(0.10, 0.08, n), 0, 1)
        accuracy      = np.clip(np.random.normal(0.62, 0.15, n), 0, 1)
        safety        = np.clip(np.random.normal(0.45, 0.20, n), 0, 1)
    else:  # instruction-tuned
        format_scores = np.clip(np.random.normal(0.88, 0.08, n), 0, 1)
        adherence     = np.clip(np.random.normal(0.85, 0.10, n), 0, 1)
        conciseness   = np.clip(np.random.normal(0.82, 0.10, n), 0, 1)
        accuracy      = np.clip(np.random.normal(0.64, 0.14, n), 0, 1)
        safety        = np.clip(np.random.normal(0.91, 0.06, n), 0, 1)

    return {
        "Format compliance": np.mean(format_scores),
        "Instruction adherence": np.mean(adherence),
        "Conciseness": np.mean(conciseness),
        "Factual accuracy": np.mean(accuracy),
        "Safety": np.mean(safety),
    }

# Evaluate 100 diverse instructions
dummy = [""] * 100

base_scores = evaluate_instruction_following(dummy, dummy, mode="base")
sft_scores = evaluate_instruction_following(dummy, dummy, mode="sft")

print(f"{'Dimension':<25} {'Base Model':>12} {'After SFT':>12} {'Change':>10}")
print("-" * 62)
for dim in base_scores:
    b, s = base_scores[dim], sft_scores[dim]
    delta = s - b
    arrow = "+" if delta > 0 else ""
    print(f"{dim:<25} {b:>11.1%} {s:>11.1%} {arrow}{delta:>9.1%}")
#
# Dimension                  Base Model    After SFT     Change
# --------------------------------------------------------------
# Format compliance              14.9%        87.7%     +72.8%
# Instruction adherence          20.6%        84.5%     +63.9%
# Conciseness                    10.0%        81.4%     +71.4%
# Factual accuracy               60.5%        63.2%      +2.7%
# Safety                         43.5%        90.8%     +47.3%
