"""
FastAPI search API with snippet highlighting.

Blog post: https://dadops.dev/blog/building-ai-search-engine/
Code Block 4.

Requires: fastapi, uvicorn, sentence-transformers.
Run with: uvicorn search_api:app --reload

NOTE: This script requires a populated search.db — run search_engine.py first.
Without API deployment, we verify the highlight_snippet function standalone.
"""
import re

from pydantic import BaseModel


# ── Code Block 4: Snippet highlighter (the logic that runs without a server) ──

def highlight_snippet(content, query, window=200):
    terms = query.lower().split()
    sentences = content.split(". ")
    best_sent, best_count = sentences[0], 0
    for sent in sentences:
        count = sum(1 for t in terms if t in sent.lower())
        if count > best_count:
            best_sent, best_count = sent, count

    snippet = best_sent[:window].strip()
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        snippet = pattern.sub(f"<mark>{term}</mark>", snippet)
    return snippet


class SearchResult(BaseModel):
    title: str
    snippet: str
    score: float
    source: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


# ── Self-tests ──

def test_highlight_snippet():
    """Test the snippet highlighting function."""
    print("=== highlight_snippet Tests ===")

    content = ("Neural networks learn through backpropagation. "
               "The training loop computes gradients for each layer. "
               "Gradient descent updates the weights to minimize loss.")

    # Test 1: Single term
    snippet = highlight_snippet(content, "gradient", window=200)
    assert "<mark>gradient</mark>" in snippet.lower()
    print(f"  Single term: {snippet[:80]}...")

    # Test 2: Multi-term — should pick best-matching sentence
    snippet = highlight_snippet(content, "gradient descent weights", window=200)
    assert "<mark>" in snippet
    print(f"  Multi-term: {snippet[:80]}...")

    # Test 3: No match — should still return first sentence
    snippet = highlight_snippet(content, "quantum computing", window=200)
    assert "Neural networks" in snippet
    print(f"  No match: {snippet[:80]}...")

    # Test 4: Window truncation
    snippet = highlight_snippet(content, "neural", window=30)
    assert len(snippet) <= 60  # window + markup overhead
    print(f"  Window=30: '{snippet}'")

    print("  PASS\n")


def test_pydantic_models():
    """Test Pydantic response models."""
    print("=== Pydantic Model Tests ===")

    result = SearchResult(
        title="Test Doc",
        snippet="<mark>test</mark> content",
        score=0.9847,
        source="test.html"
    )
    assert result.score == 0.9847
    print(f"  SearchResult: {result.title} (score: {result.score})")

    response = SearchResponse(
        query="test query",
        results=[result],
        total=1
    )
    assert response.total == 1
    assert len(response.results) == 1
    print(f"  SearchResponse: {response.query} → {response.total} result(s)")
    print("  PASS\n")


if __name__ == "__main__":
    test_highlight_snippet()
    test_pydantic_models()
    print("NOTE: FastAPI server not started — run 'uvicorn search_api:app' to serve.")
    print("All search API tests passed!")
