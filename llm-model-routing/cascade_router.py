"""
Code Block 5: Cascade router with quality check — try cheapest first, escalate on failure.

From: https://dadops.dev/blog/llm-model-routing/

Starts with the cheapest model tier and escalates if the quality check
fails. Tracks cost and attempts across tiers.

No API key required (quality_check is pure heuristics; call_llm_fn is a callback).
"""

from dataclasses import dataclass


@dataclass
class CascadeResult:
    answer: str
    tier_used: int
    attempts: int
    total_cost: float


def quality_check(query: str, response: str) -> bool:
    """Heuristic quality gate — cheap and fast."""
    # Too short? Probably a bad answer
    if len(response.split()) < 10 and len(query.split()) > 15:
        return False
    # Refusal or uncertainty signals
    refusal_phrases = ["i'm not sure", "i cannot", "as an ai", "i don't have"]
    if any(p in response.lower() for p in refusal_phrases):
        return False
    # For structured output: did it return valid format?
    if "json" in query.lower() and not response.strip().startswith("{"):
        return False
    # Multi-part question: check all parts addressed
    question_count = query.count("?")
    if question_count > 1:
        paragraphs = response.count("\n\n") + 1
        if paragraphs < question_count:
            return False
    return True


TIER_MODELS = {1: "haiku", 2: "sonnet", 3: "opus"}
TIER_COSTS  = {1: 0.00025, 2: 0.003, 3: 0.015}  # per 1K input tokens


def cascade_route(query: str, call_llm_fn) -> CascadeResult:
    """Try the cheapest model first, escalate if quality check fails."""
    total_cost = 0.0
    for tier in [1, 2, 3]:
        response = call_llm_fn(query, model=TIER_MODELS[tier])
        est_tokens = len(query.split()) * 1.3
        total_cost += TIER_COSTS[tier] * (est_tokens / 1000)

        if quality_check(query, response) or tier == 3:
            return CascadeResult(
                answer=response,
                tier_used=tier,
                attempts=tier,
                total_cost=total_cost
            )
    # unreachable — tier 3 always passes
    return CascadeResult(response, 3, 3, total_cost)


if __name__ == "__main__":
    # Simulate with mock LLM function
    def mock_llm(query, model="haiku"):
        if model == "haiku":
            return "I'm not sure about that."  # fails quality check
        elif model == "sonnet":
            return ("Here's a detailed analysis of the billing system's tax "
                    "handling across different states, covering nexus rules, "
                    "rate calculations, and exemption handling.")
        else:
            return "Comprehensive multi-approach analysis..."

    result = cascade_route(
        "Explain why our billing system charges tax differently in each state",
        mock_llm
    )
    print(f"Tier used: {result.tier_used}")
    print(f"Attempts: {result.attempts}")
    print(f"Total cost: ${result.total_cost:.6f}")
    print(f"Answer: {result.answer[:80]}...")
