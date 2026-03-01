"""
Level 1: Retrieval evaluation — Precision@k, Recall@k, and MRR.

Blog post: https://dadops.dev/blog/evaluating-llm-systems/
Code Block 2.

Uses a mock retriever to demonstrate the metrics. In production,
you'd plug in your actual vector search / BM25 retriever.
"""
from dataclasses import dataclass


@dataclass
class MockDoc:
    id: str


class MockRetriever:
    """A fake retriever that returns pre-defined results for testing."""

    def __init__(self, results_map: dict):
        """results_map: {query_string: [doc_id, doc_id, ...]}"""
        self.results_map = results_map

    def search(self, query: str, top_k: int = 5) -> list:
        doc_ids = self.results_map.get(query, [])[:top_k]
        return [MockDoc(id=d) for d in doc_ids]


def evaluate_retriever(retriever, test_cases, k=5):
    """Measure retrieval quality with Precision@k, Recall@k, and MRR."""
    metrics = {"precision": [], "recall": [], "mrr": []}

    for case in test_cases:
        retrieved_ids = [doc.id for doc in retriever.search(case["query"], top_k=k)]
        relevant = set(case["relevant_doc_ids"])

        # Precision@k: what fraction of retrieved docs are relevant?
        hits = sum(1 for d in retrieved_ids if d in relevant)
        metrics["precision"].append(hits / k)

        # Recall@k: what fraction of relevant docs did we find?
        metrics["recall"].append(hits / len(relevant) if relevant else 1.0)

        # MRR: how high is the first relevant result?
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant:
                metrics["mrr"].append(1.0 / rank)
                break
        else:
            metrics["mrr"].append(0.0)

    return {name: sum(vals) / len(vals) for name, vals in metrics.items()}


if __name__ == "__main__":
    print("=== Retrieval Evaluation Demo ===\n")

    # Set up mock retriever with known results
    retriever = MockRetriever({
        "What is our refund policy?": ["doc_refund", "doc_faq", "doc_shipping", "doc_pricing", "doc_about"],
        "How do I cancel my subscription?": ["doc_billing", "doc_cancel", "doc_faq", "doc_about", "doc_terms"],
        "What payment methods do you accept?": ["doc_about", "doc_faq", "doc_pricing", "doc_payment", "doc_terms"],
    })

    # Test cases with known relevant documents
    test_cases = [
        {
            "query": "What is our refund policy?",
            "relevant_doc_ids": ["doc_refund", "doc_faq"]
        },
        {
            "query": "How do I cancel my subscription?",
            "relevant_doc_ids": ["doc_cancel", "doc_billing"]
        },
        {
            "query": "What payment methods do you accept?",
            "relevant_doc_ids": ["doc_payment"]
        },
    ]

    results = evaluate_retriever(retriever, test_cases, k=5)

    print(f"Precision@5: {results['precision']:.3f}")
    print(f"Recall@5:    {results['recall']:.3f}")
    print(f"MRR:         {results['mrr']:.3f}")

    # Interpretation (blog quote):
    # "If MRR is 0.5, the first relevant doc is typically at position 2."
    # "If Recall@5 is 0.8, you're finding 80% of relevant docs in the top 5."
    print(f"\nInterpretation:")
    if results["mrr"] > 0:
        avg_pos = 1.0 / results["mrr"]
        print(f"  MRR={results['mrr']:.2f} → first relevant doc at ~position {avg_pos:.1f}")
    print(f"  Recall@5={results['recall']:.0%} → finding {results['recall']:.0%} of relevant docs in top 5")

    # Expected output:
    # Precision@5: 0.267
    # Recall@5:    0.833
    # MRR:         0.556
