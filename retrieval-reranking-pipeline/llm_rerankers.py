"""LLM-based rerankers: pointwise and listwise approaches.

Requires: openai package and OPENAI_API_KEY environment variable.
"""
import json
from openai import OpenAI
from helpers import RankedDoc


class PointwiseReranker:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def rerank(self, query: str, documents: list[dict],
               top_k: int = 10) -> list[RankedDoc]:
        scored = []
        for i, doc in enumerate(documents):
            prompt = (
                f"Rate the relevance of this document to the query.\n\n"
                f"Query: {query}\n\n"
                f"Document: {doc['text'][:500]}\n\n"
                f"Reply with ONLY a JSON object: "
                f'{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}'
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80,
            )
            try:
                result = json.loads(resp.choices[0].message.content)
                score = float(result["score"])
            except (json.JSONDecodeError, KeyError, ValueError):
                score = 0.0

            scored.append(RankedDoc(
                doc_id=doc["id"], text=doc["text"],
                score=score, original_rank=i + 1,
            ))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


class ListwiseReranker:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def rerank(self, query: str, documents: list[dict],
               top_k: int = 10) -> list[RankedDoc]:
        # Build a numbered list of document snippets
        doc_list = "\n".join(
            f"[{i+1}] {doc['text'][:200]}"
            for i, doc in enumerate(documents)
        )
        prompt = (
            f"Given this query, rank the documents by relevance.\n\n"
            f"Query: {query}\n\n"
            f"Documents:\n{doc_list}\n\n"
            f"Return ONLY a JSON array of document numbers in order "
            f"from most to least relevant, e.g. [3, 1, 7, 2, ...]. "
            f"Include ALL document numbers."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        try:
            ranking = json.loads(resp.choices[0].message.content)
            # Convert 1-indexed ranking to results
            results = []
            for rank_pos, doc_num in enumerate(ranking):
                idx = doc_num - 1
                if 0 <= idx < len(documents):
                    results.append(RankedDoc(
                        doc_id=documents[idx]["id"],
                        text=documents[idx]["text"],
                        score=1.0 - (rank_pos / len(ranking)),
                        original_rank=idx + 1,
                    ))
            return results[:top_k]
        except (json.JSONDecodeError, TypeError):
            # Fallback: return original order
            return [
                RankedDoc(doc["id"], doc["text"], 0.0, i + 1)
                for i, doc in enumerate(documents[:top_k])
            ]


print("LLM reranker classes defined successfully (requires API key to run)")
