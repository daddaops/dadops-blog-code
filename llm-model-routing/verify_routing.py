"""
Verification script for LLM Model Routing blog post.

Tests all routing implementations that don't require API keys or ML models.
"""

from heuristic_router import heuristic_route, COMPLEX_KEYWORDS, MODERATE_KEYWORDS
from cascade_router import quality_check, cascade_route, CascadeResult, TIER_COSTS, TIER_MODELS
from evaluation import RouterMetrics, evaluate_router
from production_router import ProductionRouter

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


print("=" * 60)
print("LLM Model Routing — Verification Suite")
print("=" * 60)

# --- Heuristic Router ---
print("\n--- Heuristic Router ---")

check("Simple query → Tier 1",
      heuristic_route("What time does the store close?") == 1)
check("Moderate query → Tier 2",
      heuristic_route("Summarize this customer's last 5 interactions") == 2)
check("Complex query → Tier 3",
      heuristic_route("Explain why our billing system charges tax differently in each state "
                      "and compare the three approaches to fixing it step by step") == 3)

# Edge cases
check("Empty query → Tier 1", heuristic_route("") == 1)
check("Short question → Tier 1", heuristic_route("hello?") == 1)
check("Query with code → gets code bonus",
      heuristic_route("def foo(): pass") >= 2)
check("Multi-question → higher tier",
      heuristic_route("What is X? And why does Y happen? And how does Z work?") >= 2)

# Keyword detection
check("COMPLEX_KEYWORDS is a set", isinstance(COMPLEX_KEYWORDS, set))
check("MODERATE_KEYWORDS is a set", isinstance(MODERATE_KEYWORDS, set))
check("10 complex keywords", len(COMPLEX_KEYWORDS) == 10)
check("9 moderate keywords", len(MODERATE_KEYWORDS) == 9)

# --- Quality Check ---
print("\n--- Quality Check (Cascade) ---")

check("Good response passes",
      quality_check("Simple question", "Here is a detailed answer with enough words to pass."))
check("Short response to long query fails",
      not quality_check("This is a long complex multi-word query about many things", "Yes."))
check("Refusal detected",
      not quality_check("Help me", "I'm not sure I can help with that particular request."))
check("AI refusal detected",
      not quality_check("Help me", "As an AI, I cannot provide that information to you."))
check("JSON format check passes",
      quality_check('Return json data', '{"key": "value", "count": 42}'))
check("JSON format check fails",
      not quality_check('Return json data', 'Here is the data: key=value'))
check("Multi-question coverage check",
      not quality_check("What is X? What is Y?", "X is a variable."))
check("Multi-question passes with paragraphs",
      quality_check("What is X? What is Y?", "X is a variable.\n\nY is another variable."))

# --- CascadeResult dataclass ---
print("\n--- CascadeResult ---")

cr = CascadeResult(answer="test", tier_used=2, attempts=2, total_cost=0.001)
check("CascadeResult fields", cr.answer == "test" and cr.tier_used == 2)
check("TIER_MODELS has 3 tiers", len(TIER_MODELS) == 3)
check("TIER_COSTS has 3 tiers", len(TIER_COSTS) == 3)
check("Tier 1 cheapest", TIER_COSTS[1] < TIER_COSTS[2] < TIER_COSTS[3])

# --- Cascade Route ---
print("\n--- Cascade Route ---")


def mock_llm(query, model="haiku"):
    if model == "haiku":
        return "I'm not sure about that."  # fails quality check
    elif model == "sonnet":
        return ("Here's a detailed analysis of the billing system's tax "
                "handling across different states, covering nexus rules, "
                "rate calculations, and exemption handling.")
    else:
        return "Comprehensive analysis..."


result = cascade_route("Explain why our billing system charges tax differently", mock_llm)
check("Cascade escalates past Tier 1 (haiku refused)", result.tier_used >= 2)
check("Cascade stops at Tier 2 (good sonnet response)", result.tier_used == 2)
check("Attempts = tier_used", result.attempts == result.tier_used)
check("Total cost > 0", result.total_cost > 0)

# Test with all-good responses
def good_llm(query, model="haiku"):
    return "Here is a comprehensive and detailed answer to your question about the topic."

result2 = cascade_route("What time is it?", good_llm)
check("Good Tier 1 response stays at Tier 1", result2.tier_used == 1)

# --- RouterMetrics ---
print("\n--- RouterMetrics ---")

m = RouterMetrics(name="test", correct=8, total=10, total_cost=0.003,
                  frontier_cost=0.015, quality_matches=9)
check("Accuracy = 80%", abs(m.accuracy - 0.8) < 0.001)
check("Cost savings = 80%", abs(m.cost_savings - 0.8) < 0.001)
check("Quality preservation = 90%", abs(m.quality_preservation - 0.9) < 0.001)

empty_m = RouterMetrics(name="empty")
check("Empty metrics: accuracy = 0", empty_m.accuracy == 0)
check("Empty metrics: cost_savings = 0", empty_m.cost_savings == 0)

# --- Evaluate Router ---
print("\n--- Evaluate Router ---")

test_queries = [
    "What time does the store close?",
    "Summarize this customer's last 5 interactions",
    "Explain why our billing system charges tax differently in each state "
    "and compare the three approaches to fixing it step by step",
]
oracle_labels = [1, 2, 3]

m = evaluate_router("Heuristic", heuristic_route, test_queries, oracle_labels)
check("Evaluated 3 queries", m.total == 3)
check("Heuristic accuracy >= 66%", m.accuracy >= 0.66)
check("Quality preservation >= 66%", m.quality_preservation >= 0.66)
check("Cost savings > 0% vs frontier", m.cost_savings > 0)

# --- ProductionRouter ---
print("\n--- ProductionRouter ---")


class MockProvider:
    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail
    def complete(self, query):
        if self.fail:
            raise ConnectionError(f"{self.name} is down")
        return f"[{self.name}] Response"
    def __repr__(self):
        return self.name
    def __hash__(self):
        return hash(self.name)


providers = {
    1: [MockProvider("haiku-a"), MockProvider("haiku-b")],
    2: [MockProvider("sonnet-a")],
    3: [MockProvider("opus-a")],
}

router = ProductionRouter(heuristic_route, providers)
result = router.route("What time is it?")
check("Simple query routed successfully", "haiku-a" in result)
check("Stats tracked total", router.stats["total"] == 1)
check("Stats tracked tier distribution", router.stats["tier_dist"][1] == 1)

# Test fallback
providers_fail = {
    1: [MockProvider("haiku-a", fail=True), MockProvider("haiku-b", fail=True)],
    2: [MockProvider("sonnet-a")],
    3: [MockProvider("opus-a")],
}
router2 = ProductionRouter(heuristic_route, providers_fail)
result = router2.route("What time is it?")
check("Fallback to Tier 2 on Tier 1 failure", "sonnet-a" in result)
check("Fallback count = 1", router2.stats["fallbacks"] == 1)

# Test circuit breaker
providers_circuit = {
    1: [MockProvider("haiku-fail", fail=True)],
    2: [MockProvider("sonnet-ok")],
    3: [MockProvider("opus-ok")],
}
router3 = ProductionRouter(heuristic_route, providers_circuit)
for _ in range(5):
    try:
        router3.route("simple question")
    except RuntimeError:
        pass  # may fail if all providers exhausted during warmup
provider_ref = providers_circuit[1][0]
check("Circuit opens after 5 failures", provider_ref in router3.circuit_open)

# --- Summary ---
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)

if failed > 0:
    exit(1)
