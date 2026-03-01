"""
Code Block 1: Heuristic keyword-based query router.

From: https://dadops.dev/blog/llm-model-routing/

Routes queries to model tiers (1=cheap, 2=moderate, 3=expensive) using
keyword matching, word count, question count, and code detection heuristics.

No dependencies required.
"""

import re

# Signals that suggest the query needs more reasoning power
COMPLEX_KEYWORDS = {
    "explain", "compare", "analyze", "why does", "step by step",
    "trade-off", "debug", "evaluate", "pros and cons", "design"
}
MODERATE_KEYWORDS = {
    "summarize", "rewrite", "generate", "translate", "convert",
    "write a", "draft", "list the", "what are the"
}


def heuristic_route(query: str) -> int:
    """Route a query to a model tier (1, 2, or 3) using surface heuristics."""
    text = query.lower().strip()
    word_count = len(text.split())

    # Multi-part questions are harder
    question_marks = text.count("?")
    has_code = bool(re.search(r"```|def |class |function |SELECT ", query))

    # Score complexity 0-10
    score = 0
    score += min(word_count // 20, 3)          # longer queries are harder
    score += question_marks - 1 if question_marks > 1 else 0
    score += 2 if has_code else 0

    for kw in COMPLEX_KEYWORDS:
        if kw in text:
            score += 2
            break
    for kw in MODERATE_KEYWORDS:
        if kw in text:
            score += 1
            break

    if score >= 5:
        return 3
    elif score >= 2:
        return 2
    return 1


if __name__ == "__main__":
    # Quick test
    queries = [
        "What time does the store close?",                  # simple lookup
        "Summarize this customer's last 5 interactions",    # moderate
        "Explain why our billing system charges tax differently in each state "
        "and compare the three approaches to fixing it step by step",  # complex
    ]
    for q in queries:
        print(f"Tier {heuristic_route(q)}: {q[:60]}...")

    # Expected output:
    # Tier 1: What time does the store close?...
    # Tier 1: Summarize this customer's last 5 interactions...
    # Tier 2: Explain why our billing system charges tax differently in ea...
