"""
Verify the Fine-Tuning ROI Calculator math from the blog's interactive demo.

Blog post: https://dadops.dev/blog/fine-tuning-llms/
Interactive Demo JS (lines 787-848).

Reproduces the JavaScript calculator logic in Python to verify the math.
No API key needed.

Blog claims (at default slider values):
  - At 10K requests/month, 1,800 few-shot tokens, 200 training examples:
  - Cost reduction from $3.30 to $0.66 per month (the prose says "15x input token reduction")
  - Training cost ~$1.20 (prose claim) vs calculator output
"""

# GPT-4o-mini pricing (matching JS constants)
INPUT_COST = 0.15 / 1e6     # $0.15 per 1M input tokens
OUTPUT_COST = 0.60 / 1e6    # $0.60 per 1M output tokens
TRAINING_COST = 3.00 / 1e6  # $3.00 per 1M training tokens
AVG_OUTPUT_TOKENS = 80
FT_SYSTEM_TOKENS = 40       # short system prompt after fine-tuning
AVG_USER_TOKENS = 80
EPOCHS = 3


def calculate_roi(requests, fewshot_tokens, training_examples):
    """Replicate the blog's ROI calculator logic."""
    # Base model cost: few-shot examples + user input + output
    base_input_per_req = fewshot_tokens + AVG_USER_TOKENS
    base_monthly = requests * (base_input_per_req * INPUT_COST + AVG_OUTPUT_TOKENS * OUTPUT_COST)

    # Fine-tuned model cost: short system prompt + user input + output (no few-shot)
    ft_input_per_req = FT_SYSTEM_TOKENS + AVG_USER_TOKENS
    ft_monthly = requests * (ft_input_per_req * INPUT_COST + AVG_OUTPUT_TOKENS * OUTPUT_COST)

    # Training cost
    avg_tokens_per_example = 300  # system + user + assistant
    total_training_tokens = training_examples * avg_tokens_per_example * EPOCHS
    training_cost = total_training_tokens * TRAINING_COST

    # Break-even
    monthly_savings = base_monthly - ft_monthly
    if monthly_savings <= 0:
        breakeven_days = float('inf')
    else:
        breakeven_days = (training_cost / monthly_savings) * 30

    return {
        "base_monthly": base_monthly,
        "ft_monthly": ft_monthly,
        "training_cost": training_cost,
        "monthly_savings": monthly_savings,
        "breakeven_days": breakeven_days,
        "input_token_ratio": base_input_per_req / ft_input_per_req,
    }


if __name__ == "__main__":
    print("=== Fine-Tuning ROI Calculator Verification ===\n")

    # Test with blog's default slider values
    result = calculate_roi(
        requests=10_000,
        fewshot_tokens=1_800,
        training_examples=200
    )

    print("Default values (10K requests, 1800 few-shot tokens, 200 examples):")
    print(f"  Base monthly cost:      ${result['base_monthly']:.2f}")
    print(f"  Fine-tuned monthly:     ${result['ft_monthly']:.2f}")
    print(f"  Training cost (one-time): ${result['training_cost']:.2f}")
    print(f"  Monthly savings:        ${result['monthly_savings']:.2f}")
    print(f"  Break-even:             {result['breakeven_days']:.0f} days")
    print(f"  Input token ratio:      {result['input_token_ratio']:.1f}x")

    # Verify blog's specific claims
    print("\n--- Claim Verification ---")

    # Blog claim: "15x input token reduction"
    ratio = (1800 + 80) / (40 + 80)
    print(f"Input token ratio: {ratio:.1f}x (blog says 15x)")

    # Blog claim: "$3.30 to $0.66 at 10K requests/month"
    print(f"Base cost: ${result['base_monthly']:.2f} (blog says ~$3.30)")
    print(f"FT cost:   ${result['ft_monthly']:.2f} (blog says ~$0.66)")

    # Blog claim: "Training cost ~$0.54 for 200 examples"
    # Blog says "~300 tokens each, 3 epochs" — matches calculator
    calc_training = result['training_cost']
    print(f"Training cost (calculator: 300 tok, 3 epochs): ${calc_training:.2f}")

    # Blog claim: cost per request comparison
    base_per_req = result['base_monthly'] / 10_000
    ft_per_req = result['ft_monthly'] / 10_000
    print(f"\nCost per request - Base: ${base_per_req:.5f} (blog says ~$0.0003)")
    print(f"Cost per request - FT:   ${ft_per_req:.6f} (blog says ~$0.00007)")

    # Expected output:
    # Base monthly cost: ~$3.30
    # Fine-tuned monthly: ~$0.66
    # Training cost: $0.54 (300 tok × 3 epochs)
