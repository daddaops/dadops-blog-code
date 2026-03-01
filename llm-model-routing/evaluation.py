"""
Code Block 6: Router evaluation framework.

From: https://dadops.dev/blog/llm-model-routing/

Benchmarks routing strategies against oracle labels (cheapest correct tier).
Measures accuracy, cost savings, and quality preservation.

No API key required.
"""

from dataclasses import dataclass
from collections import defaultdict
from heuristic_router import heuristic_route

TIER_COSTS = {1: 0.00025, 2: 0.003, 3: 0.015}  # per 1K input tokens


@dataclass
class RouterMetrics:
    name: str
    correct: int = 0
    total: int = 0
    total_cost: float = 0.0
    frontier_cost: float = 0.0
    quality_matches: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0

    @property
    def cost_savings(self) -> float:
        return 1 - (self.total_cost / self.frontier_cost) if self.frontier_cost else 0

    @property
    def quality_preservation(self) -> float:
        return self.quality_matches / self.total if self.total else 0


def evaluate_router(name, router_fn, test_set, oracle_tiers):
    """Benchmark a router against the oracle (cheapest correct tier)."""
    metrics = RouterMetrics(name=name)

    for query, oracle_tier in zip(test_set, oracle_tiers):
        predicted_tier = router_fn(query)
        metrics.total += 1
        metrics.correct += int(predicted_tier == oracle_tier)
        metrics.total_cost += TIER_COSTS[predicted_tier]
        metrics.frontier_cost += TIER_COSTS[3]
        # Quality preserved if predicted tier >= oracle tier
        metrics.quality_matches += int(predicted_tier >= oracle_tier)

    return metrics


if __name__ == "__main__":
    # Test with heuristic router only (no API key or ML model needed)
    test_queries = [
        "What time does the store close?",
        "Summarize this customer's last 5 interactions",
        "Explain why our billing system charges tax differently in each state "
        "and compare the three approaches to fixing it step by step",
        "What is the return policy?",
        "Write a professional email to a client",
    ]
    oracle_labels = [1, 2, 3, 1, 2]

    m = evaluate_router("Heuristic", heuristic_route, test_queries, oracle_labels)
    print(f"{m.name:<20} Acc={m.accuracy:.0%}  Save={m.cost_savings:.0%}  "
          f"Qual={m.quality_preservation:.0%}")
