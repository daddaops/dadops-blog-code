"""Feature-based reranker: weighted linear combination of retrieval signals."""
import math
from collections import Counter
from helpers import RankedDoc, SAMPLE_CANDIDATES, SAMPLE_QUERY


class FeatureReranker:
    def __init__(self, weights=None):
        # Default weights learned via grid search on a dev set
        self.weights = weights or {
            "bm25_score":       0.30,
            "vector_sim":       0.25,
            "query_coverage":   0.20,
            "recency":          0.10,
            "doc_length_norm":  0.08,
            "source_authority": 0.07,
        }

    def extract_features(self, query: str, doc: dict) -> dict:
        q_terms = set(query.lower().split())
        d_terms = Counter(doc["text"].lower().split())

        # Term coverage: fraction of query terms found in document
        covered = sum(1 for t in q_terms if d_terms[t] > 0)
        coverage = covered / max(len(q_terms), 1)

        # Recency score: exponential decay from document age in days
        age_days = doc.get("age_days", 365)
        recency = math.exp(-age_days / 730)  # half-life ~2 years

        # Length normalization: penalize very short or very long docs
        word_count = sum(d_terms.values())
        ideal_length = 300  # target chunk size in words
        length_norm = 1.0 - min(abs(word_count - ideal_length) / 1000, 1.0)

        return {
            "bm25_score":       doc.get("bm25_score", 0.0),
            "vector_sim":       doc.get("vector_sim", 0.0),
            "query_coverage":   coverage,
            "recency":          recency,
            "doc_length_norm":  length_norm,
            "source_authority": doc.get("authority", 0.5),
        }

    def rerank(self, query: str, documents: list[dict],
               top_k: int = 10) -> list[RankedDoc]:
        scored = []
        for i, doc in enumerate(documents):
            features = self.extract_features(query, doc)
            # Weighted linear combination of all features
            composite = sum(
                self.weights[k] * features[k] for k in self.weights
            )
            scored.append(RankedDoc(
                doc_id=doc["id"], text=doc["text"],
                score=composite, original_rank=i + 1,
            ))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


# Smoke test with sample data
reranker = FeatureReranker()
results = reranker.rerank(SAMPLE_QUERY, SAMPLE_CANDIDATES, top_k=5)
print("Feature-based reranking results:")
for r in results:
    print(f"  {r.doc_id}: score={r.score:.4f} (was rank {r.original_rank})")
