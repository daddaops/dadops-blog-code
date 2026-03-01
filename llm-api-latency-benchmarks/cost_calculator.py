"""LLM Effective Cost Calculator

Calculates the true cost of LLM API usage accounting for retries, timeouts,
and error rates — not just the per-token price on the pricing page.

No API keys needed — pure math.

Blog post: https://dadops.co/blog/llm-api-latency-benchmarks/
Code Block 5 from the blog.
"""


def calculate_effective_cost(
    base_input_price: float,   # $/1M input tokens
    base_output_price: float,  # $/1M output tokens
    avg_input_tokens: int,
    avg_output_tokens: int,
    error_rate: float,         # 0.0 to 1.0
    timeout_rate: float,       # 0.0 to 1.0
    requests_per_day: int,
) -> dict:
    """Calculate the true monthly cost of an LLM API provider."""
    # Base cost per request
    input_cost = (avg_input_tokens / 1_000_000) * base_input_price
    output_cost = (avg_output_tokens / 1_000_000) * base_output_price
    base_per_request = input_cost + output_cost

    # Retry overhead: each error means re-sending the request
    # Errors still consume input tokens on most providers
    retry_multiplier = 1 + error_rate  # ~1.02 for 2% error rate

    # Timeout overhead: timed-out requests consumed tokens but produced
    # no usable output — pure waste
    timeout_multiplier = 1 / (1 - timeout_rate)  # ~1.01 for 1% timeout

    effective_per_request = (
        base_per_request * retry_multiplier * timeout_multiplier
    )

    daily_cost = effective_per_request * requests_per_day
    monthly_cost = daily_cost * 30

    return {
        "base_per_request": base_per_request,
        "effective_per_request": effective_per_request,
        "overhead_pct": (effective_per_request / base_per_request - 1) * 100,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
    }


if __name__ == "__main__":
    print("LLM Effective Cost Calculator")
    print("=" * 80)
    print(f"Scenario: 100K requests/day, 500 input + 200 output tokens each\n")

    # Example: 100K requests/day, 500 input + 200 output tokens each
    models = [
        ("GPT-4o",         2.50, 10.00, 0.032, 0.008),
        ("GPT-4o-mini",    0.15,  0.60, 0.008, 0.002),
        ("Claude Sonnet",  3.00, 15.00, 0.015, 0.005),
        ("Claude Haiku",   0.25,  1.25, 0.004, 0.001),
    ]

    print(f"{'Model':20s}  {'Base $/Kreq':>12}  {'Eff $/Kreq':>12}  "
          f"{'Overhead':>10}  {'Monthly':>10}")
    print("-" * 72)

    for name, inp, out, err, tout in models:
        result = calculate_effective_cost(
            inp, out, 500, 200, err, tout, 100_000
        )
        print(f"{name:20s}  ${result['base_per_request']*1000:>10.3f}  "
              f"${result['effective_per_request']*1000:>10.3f}  "
              f"{result['overhead_pct']:>9.1f}%  "
              f"${result['monthly_cost']:>9,.0f}")

    # Verify against blog claims
    print("\n" + "=" * 80)
    print("BLOG CLAIMS VERIFICATION")
    print("=" * 80)

    blog_claims = {
        "GPT-4o": {"base": 0.00325, "effective": 0.00339, "overhead": 4.1, "monthly": 10170},
        "GPT-4o-mini": {"base": 0.000195, "effective": 0.000197, "overhead": 1.0, "monthly": 591},
        "Claude Sonnet": {"base": 0.00450, "effective": 0.00460, "overhead": 2.0, "monthly": 13800},
        "Claude Haiku": {"base": 0.000375, "effective": 0.000377, "overhead": 0.5, "monthly": 1131},
    }

    for name, inp, out, err, tout in models:
        result = calculate_effective_cost(inp, out, 500, 200, err, tout, 100_000)
        claims = blog_claims[name]

        base_match = abs(result["base_per_request"] - claims["base"]) < 0.00001
        eff_match = abs(result["effective_per_request"] - claims["effective"]) < 0.0001
        oh_match = abs(result["overhead_pct"] - claims["overhead"]) < 0.5
        monthly_match = abs(result["monthly_cost"] - claims["monthly"]) < 100

        print(f"\n{name}:")
        print(f"  Base:      code={result['base_per_request']:.6f}  blog={claims['base']:.6f}  "
              f"{'MATCH' if base_match else 'MISMATCH'}")
        print(f"  Effective: code={result['effective_per_request']:.6f}  blog={claims['effective']:.6f}  "
              f"{'MATCH' if eff_match else 'MISMATCH'}")
        print(f"  Overhead:  code={result['overhead_pct']:.1f}%  blog={claims['overhead']:.1f}%  "
              f"{'MATCH' if oh_match else 'MISMATCH'}")
        print(f"  Monthly:   code=${result['monthly_cost']:,.0f}  blog=${claims['monthly']:,}  "
              f"{'MATCH' if monthly_match else 'MISMATCH'}")
