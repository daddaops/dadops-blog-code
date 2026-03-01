"""
Cost-Aware Model Router — routes requests to cheapest adequate model tier.

From: https://dadops.dev/blog/llm-cost-optimization/
Code Block 3: "Model Routing — Right-Size Every Request"

Uses keyword + length heuristics to classify complexity.
Blog claims: 87% correct on 500 requests, 78% cost reduction, <3% quality loss.
Blog claims blended input $3.00/M (80% reduction from all-Opus $15/M).
"""

from dataclasses import dataclass
from enum import Enum


class Tier(Enum):
    FAST = "haiku"      # simple factual, classification, extraction
    BALANCED = "sonnet"  # moderate reasoning, summarization
    POWERFUL = "opus"    # complex analysis, multi-step reasoning


TIER_PRICING = {  # (input_per_M, output_per_M)
    Tier.FAST:     (1.00,  5.00),
    Tier.BALANCED: (3.00, 15.00),
    Tier.POWERFUL: (15.00, 75.00),
}

# Complexity signals — cheap to compute, surprisingly effective
COMPLEX_KEYWORDS = {
    "analyze", "compare", "contrast", "evaluate", "synthesize",
    "implications", "tradeoffs", "recommend", "strategy", "debug",
}
SIMPLE_KEYWORDS = {
    "what is", "define", "list", "extract", "classify", "translate",
    "summarize this", "yes or no", "true or false",
}


@dataclass
class RoutingDecision:
    tier: Tier
    reason: str
    estimated_cost: float


def route_request(prompt: str, input_tokens: int,
                  est_output_tokens: int = 500) -> RoutingDecision:
    """Route a request to the cheapest model that can handle it."""
    prompt_lower = prompt.lower()
    word_count = len(prompt.split())

    # Rule 1: Short, simple queries go to fast tier
    if (word_count < 50
            and any(kw in prompt_lower for kw in SIMPLE_KEYWORDS)):
        tier = Tier.FAST
        reason = "short + simple keyword match"

    # Rule 2: Complex keywords or long prompts need more power
    elif (sum(kw in prompt_lower for kw in COMPLEX_KEYWORDS) >= 2
          or word_count > 500):
        tier = Tier.POWERFUL
        reason = "complex keywords or lengthy prompt"

    # Rule 3: Everything else goes to balanced tier
    else:
        tier = Tier.BALANCED
        reason = "moderate complexity"

    pricing = TIER_PRICING[tier]
    cost = (input_tokens * pricing[0]
            + est_output_tokens * pricing[1]) / 1_000_000

    return RoutingDecision(tier, reason, cost)


if __name__ == "__main__":
    # Example from the blog: route 5 requests and compare costs
    requests = [
        ("What is the return policy?", 120),
        ("Summarize this meeting transcript.", 3400),
        ("Analyze these 3 contracts for conflicting liability clauses "
         "and recommend amendments with tradeoffs.", 8200),
        ("Classify this support ticket: urgent or normal.", 85),
        ("Compare Q3 and Q4 revenue and evaluate growth strategy.", 2100),
    ]

    print("=== Model Routing Demo ===")
    print(f"{'Request':<55} {'Tier':<10} {'Reason':<35} {'Cost':>10}")
    print("-" * 115)

    total_cost = 0.0
    for prompt, tokens in requests:
        decision = route_request(prompt, tokens)
        total_cost += decision.estimated_cost
        print(f"  {prompt[:52]:<55} {decision.tier.value:<10} "
              f"{decision.reason:<35} ${decision.estimated_cost:.6f}")

    opus_cost = sum(
        (t * 15 + 500 * 75) / 1e6 for _, t in requests
    )

    print(f"\n  Routed: ${total_cost:.4f}  |  All-Opus: ${opus_cost:.4f}")
    print(f"  Savings: {(1 - total_cost/opus_cost)*100:.0f}%")

    # Verify blended pricing claim from blog
    # Blog claims: 60/30/10 split yields blended input $3.00/M (80% reduction)
    blended_input = 0.60 * 1.00 + 0.30 * 3.00 + 0.10 * 15.00
    blended_output = 0.60 * 5.00 + 0.30 * 15.00 + 0.10 * 75.00
    print(f"\n=== Blended Pricing (60/30/10 split) ===")
    print(f"  Blended input:  ${blended_input:.2f}/M  (blog claims: $3.00/M)")
    print(f"  Blended output: ${blended_output:.2f}/M  (blog claims: $15.00/M)")
    reduction = (1 - blended_input / 15.00) * 100
    print(f"  Input reduction: {reduction:.0f}% (blog claims: 80%)")
