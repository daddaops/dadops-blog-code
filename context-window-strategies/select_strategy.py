"""Strategy selector for context window management.

Picks the best context strategy based on document size, count, task type,
and latency budget.
"""


def select_strategy(total_tokens, num_documents=1,
                    task_type="qa", latency_budget_ms=10000):
    """Pick the best context strategy for the situation."""

    # Small enough to fit directly? Just send it.
    if total_tokens < 50_000 and num_documents == 1:
        return "direct", "Fits in context, no processing needed"

    # Multiple documents always benefit from map-reduce
    if num_documents > 1:
        return "map_reduce", f"{num_documents} docs: parallel map then reduce"

    # Single very large document
    if total_tokens > 500_000:
        if task_type == "overview":
            return "hierarchical", "Huge doc + overview: recursive summarization"
        return "agentic", "Huge doc + specific task: selective reading"

    # Medium documents (50K-500K)
    if total_tokens > 100_000:
        if latency_budget_ms < 5000:
            return "smart_truncation", "Tight latency: fast truncation"
        return "chunk_summarize", "Mid-size doc: chunk and summarize"

    # 50K-100K range
    if task_type in ("comparison", "analysis"):
        return "chunk_summarize", "Analysis needs full coverage"

    return "smart_truncation", "Default for moderate single docs"


if __name__ == "__main__":
    print("=== Strategy Selector Demo ===\n")

    # Example from the blog post
    strategy, reason = select_strategy(
        total_tokens=250_000, num_documents=1, task_type="qa"
    )
    print(f"Strategy: {strategy}")
    print(f"Reason: {reason}")
    print()

    # Test a range of scenarios
    scenarios = [
        {"total_tokens": 10_000, "num_documents": 1, "task_type": "qa"},
        {"total_tokens": 75_000, "num_documents": 1, "task_type": "analysis"},
        {"total_tokens": 75_000, "num_documents": 1, "task_type": "qa"},
        {"total_tokens": 200_000, "num_documents": 1, "task_type": "qa", "latency_budget_ms": 3000},
        {"total_tokens": 200_000, "num_documents": 1, "task_type": "qa", "latency_budget_ms": 15000},
        {"total_tokens": 100_000, "num_documents": 5, "task_type": "comparison"},
        {"total_tokens": 1_000_000, "num_documents": 1, "task_type": "overview"},
        {"total_tokens": 1_000_000, "num_documents": 1, "task_type": "qa"},
    ]

    for s in scenarios:
        strategy, reason = select_strategy(**s)
        tokens_str = f"{s['total_tokens']:>10,}"
        docs_str = f"{s['num_documents']} doc{'s' if s['num_documents'] > 1 else ' '}"
        task_str = f"{s.get('task_type', 'qa'):>10}"
        latency = s.get('latency_budget_ms', 10000)
        print(f"  {tokens_str} tokens | {docs_str} | {task_str} | {latency:>6}ms "
              f"-> {strategy:>20} | {reason}")

    print("\nDone.")
