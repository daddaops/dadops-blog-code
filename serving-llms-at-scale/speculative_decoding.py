"""
Speculative Decoding Simulator — draft-verify pipeline cost analysis.

Blog post: https://dadops.dev/blog/serving-llms-at-scale/
Code Block 4 from "Serving LLMs at Scale: From Naive to vLLM"
"""
import random

def simulate_speculative_decoding(
    total_tokens, draft_k, acceptance_rate,
    draft_cost, target_cost, seed=42
):
    """
    Simulate speculative decoding vs standard autoregressive generation.
    Costs are in arbitrary time units per forward pass.
    """
    rng = random.Random(seed)

    # Standard autoregressive: one target forward pass per token
    standard_time = total_tokens * target_cost
    standard_passes = total_tokens

    # Speculative: draft K tokens, verify in one target pass
    spec_tokens = 0
    spec_time = 0.0
    spec_target_passes = 0
    spec_draft_passes = 0

    while spec_tokens < total_tokens:
        # Draft phase: K forward passes through small model
        spec_time += draft_k * draft_cost
        spec_draft_passes += draft_k

        # Verify phase: one forward pass through target model
        spec_time += target_cost
        spec_target_passes += 1

        # Count accepted tokens (each has acceptance_rate probability)
        accepted = 0
        for i in range(draft_k):
            if rng.random() < acceptance_rate:
                accepted += 1
            else:
                break  # First rejection stops the chain
        # Always get at least 1 token (from the target model's correction)
        tokens_this_round = accepted + 1
        spec_tokens += tokens_this_round

    speedup = standard_time / spec_time
    tokens_per_pass = spec_tokens / spec_target_passes

    print(f"Speculative Decoding (K={draft_k}, accept={acceptance_rate:.0%})")
    print(f"  Draft cost: {draft_cost}, Target cost: {target_cost}")
    print(f"  Tokens generated: {spec_tokens}")
    print(f"  Target passes:    {spec_target_passes} (vs {standard_passes} standard)")
    print(f"  Tokens/target pass: {tokens_per_pass:.1f}")
    print(f"  Speedup: {speedup:.2f}x")
    return speedup


if __name__ == "__main__":
    print("=== Varying acceptance rates ===")
    speedups = []
    rates = [0.5, 0.6, 0.7, 0.8, 0.9]
    for rate in rates:
        s = simulate_speculative_decoding(
            total_tokens=200, draft_k=5,
            acceptance_rate=rate,
            draft_cost=0.05, target_cost=1.0
        )
        speedups.append(s)
        print()

    print("Summary:")
    print(f"  {'Rate':<12} {'Speedup':<10}")
    for rate, s in zip(rates, speedups):
        bar = '#' * int(s * 10)
        print(f"  {rate:<12.0%} {s:<10.2f} {bar}")
