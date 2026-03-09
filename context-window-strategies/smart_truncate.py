"""Smart priority-based truncation for context window management.

Scores document sections by relevance, position, and query overlap,
then greedily packs the highest-scoring sections into a token budget.
"""


def smart_truncate(sections, max_tokens=4096, query=None):
    """Keep the most relevant sections within a token budget.

    Each section: {"text": str, "position": float 0-1, "priority": float 0-1}
    """
    scored = []
    for sec in sections:
        score = sec.get("priority", 0.5)
        pos = sec["position"]

        # Boost beginning (context setup) and end (recency)
        if pos < 0.1:
            score += 0.3
        elif pos > 0.9:
            score += 0.2

        # Boost sections matching the query
        if query:
            query_words = set(query.lower().split())
            text_words = set(sec["text"].lower().split())
            overlap = len(query_words & text_words)
            score += min(overlap * 0.1, 0.4)

        scored.append((score, sec))

    # Sort by score descending, greedily fill the budget
    scored.sort(key=lambda x: -x[0])
    kept, budget = [], max_tokens

    for score, sec in scored:
        tokens = int(len(sec["text"].split()) * 1.3)  # ~1.3 tokens/word
        if tokens <= budget:
            kept.append(sec)
            budget -= tokens

    # Restore original order for coherent reading
    kept.sort(key=lambda s: s["position"])
    return "\n\n".join(s["text"] for s in kept)


if __name__ == "__main__":
    # Demo with sample document sections
    sections = [
        {"text": "Introduction: This contract governs the terms of service between "
                 "Party A and Party B for the delivery of software consulting services.",
         "position": 0.0, "priority": 0.8},
        {"text": "Section 1: Definitions. 'Services' means the consulting work described "
                 "in Appendix A. 'Deliverables' means all work products.",
         "position": 0.1, "priority": 0.6},
        {"text": "Section 2: Payment terms. Party A shall pay Party B within 30 days of "
                 "receiving each invoice. Late payments accrue 1.5% monthly interest.",
         "position": 0.3, "priority": 0.7},
        {"text": "Section 3: Intellectual property. All deliverables become the property "
                 "of Party A upon full payment. Party B retains rights to pre-existing tools.",
         "position": 0.5, "priority": 0.5},
        {"text": "Section 4: Confidentiality. Both parties agree to keep proprietary "
                 "information confidential for 3 years after contract termination.",
         "position": 0.6, "priority": 0.4},
        {"text": "Section 5: Liability cap. Total liability shall not exceed the fees "
                 "paid under this contract in the 12 months preceding the claim.",
         "position": 0.7, "priority": 0.6},
        {"text": "Section 6: Termination. Either party may terminate with 30 days written "
                 "notice. Fees for completed work remain due upon termination.",
         "position": 0.85, "priority": 0.5},
        {"text": "Section 7: Governing law. This contract is governed by the laws of "
                 "the State of Delaware. Disputes resolved by binding arbitration.",
         "position": 0.95, "priority": 0.3},
    ]

    print("=== Smart Truncation Demo ===\n")

    # Test 1: No query, generous budget
    result = smart_truncate(sections, max_tokens=200)
    print(f"Budget=200 tokens (no query):\n{result}\n")

    # Test 2: With a query about payment
    result = smart_truncate(sections, max_tokens=150, query="payment terms invoice")
    print(f"Budget=150 tokens (query='payment terms invoice'):\n{result}\n")

    # Test 3: Very tight budget
    result = smart_truncate(sections, max_tokens=50, query="liability")
    print(f"Budget=50 tokens (query='liability'):\n{result}\n")

    print("Done.")
