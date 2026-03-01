"""
Code Block 2: Diverse Example Selection + Strategy Comparison.

From: https://dadops.dev/blog/prompt-engineering-systematic/

Greedy max-distance algorithm for selecting maximally diverse few-shot
examples using SequenceMatcher. The selection algorithm runs standalone.

The claimed accuracy results (82%/88%/94%) require LLM API calls and
cannot be reproduced here — they are marked as SKIP.

No external dependencies required (stdlib only).
"""

from difflib import SequenceMatcher
import random


def select_diverse_examples(pool: list[dict], k: int) -> list[dict]:
    """Select k maximally diverse examples using greedy max-distance.

    Each example in pool is a dict with 'input', 'category', 'output'.
    Uses SequenceMatcher for string distance — simple but effective.
    """
    if k >= len(pool):
        return pool[:]

    # Start with the first example (arbitrary seed)
    selected = [pool[0]]
    remaining = pool[1:]

    while len(selected) < k:
        best_candidate = None
        best_min_distance = -1

        for candidate in remaining:
            # Find minimum distance from candidate to any selected example
            min_dist = min(
                1 - SequenceMatcher(
                    None,
                    candidate["input"],
                    sel["input"]
                ).ratio()
                for sel in selected
            )
            # We want the candidate that is farthest from its nearest neighbor
            if min_dist > best_min_distance:
                best_min_distance = min_dist
                best_candidate = candidate

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


if __name__ == "__main__":
    print("=== Diverse Example Selection Demo ===\n")

    # Compare three selection strategies
    example_pool = [
        {"input": "I was charged twice for order #4821", "category": "billing",
         "output": "Category: billing\nIssue: Double charge on order #4821"},
        {"input": "My package says delivered but I never got it", "category": "shipping",
         "output": "Category: shipping\nIssue: Package marked delivered but not received"},
        {"input": "How do I change my password?", "category": "account",
         "output": "Category: account\nIssue: Password change request"},
        {"input": "The app crashes when I try to checkout and then I got charged anyway",
         "category": "technical",
         "output": "Category: technical\nIssue: App crash during checkout with erroneous charge"},
        {"input": "I want to return this but I also need a refund for the shipping",
         "category": "returns",
         "output": "Category: returns\nIssue: Return request with shipping refund"},
    ]

    random.seed(42)

    # Strategy A: Random 3
    strategy_a = random.sample(example_pool, 3)
    print("Strategy A (random):")
    for ex in strategy_a:
        print(f"  [{ex['category']}] {ex['input']}")

    # Strategy B: One per category (stratified)
    seen_cats = set()
    strategy_b = []
    for ex in example_pool:
        if ex["category"] not in seen_cats and len(strategy_b) < 3:
            strategy_b.append(ex)
            seen_cats.add(ex["category"])
    print("\nStrategy B (stratified):")
    for ex in strategy_b:
        print(f"  [{ex['category']}] {ex['input']}")

    # Strategy C: Diversity-maximized
    strategy_c = select_diverse_examples(example_pool, 3)
    print("\nStrategy C (diverse):")
    for ex in strategy_c:
        print(f"  [{ex['category']}] {ex['input']}")

    # Compute pairwise distances for strategy C to show diversity
    print("\nDiversity scores (avg min-distance from neighbors):")
    for name, strat in [("A random", strategy_a), ("B stratified", strategy_b), ("C diverse", strategy_c)]:
        if len(strat) < 2:
            continue
        dists = []
        for i, ex in enumerate(strat):
            for j, other in enumerate(strat):
                if i != j:
                    dists.append(1 - SequenceMatcher(None, ex["input"], other["input"]).ratio())
        avg_dist = sum(dists) / len(dists)
        print(f"  {name}: avg pairwise distance = {avg_dist:.3f}")

    print("\nNote: Claimed accuracy results (82%/88%/94%) require LLM API calls — SKIP")
