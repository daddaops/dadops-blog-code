"""Cross-encoder reranker using sentence-transformers."""
from sentence_transformers import CrossEncoder
from helpers import RankedDoc, SAMPLE_CANDIDATES, SAMPLE_QUERY


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[dict],
               top_k: int = 10) -> list[RankedDoc]:
        """Rerank documents using cross-encoder relevance scoring.

        Args:
            query: The search query string
            documents: List of dicts with 'id' and 'text' keys
            top_k: Number of top results to return
        """
        # Form query-document pairs for the cross-encoder
        pairs = [(query, doc["text"]) for doc in documents]

        # Score all pairs in a single batch — the model handles
        # tokenization, padding, and forward pass internally
        scores = self.model.predict(pairs)

        # Combine scores with original metadata
        ranked = [
            RankedDoc(
                doc_id=doc["id"],
                text=doc["text"],
                score=float(score),
                original_rank=i + 1,
            )
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]

        # Sort by cross-encoder score (descending) and return top_k
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

# Usage
reranker = CrossEncoderReranker()
candidates = SAMPLE_CANDIDATES
results = reranker.rerank(SAMPLE_QUERY, candidates, top_k=5)
for r in results:
    print(f"  {r.doc_id}: score={r.score:.4f} (was rank {r.original_rank})")
