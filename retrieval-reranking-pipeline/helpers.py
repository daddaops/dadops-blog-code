"""Shared helpers for retrieval-reranking scripts."""
from dataclasses import dataclass


@dataclass
class RankedDoc:
    doc_id: str
    text: str
    score: float
    original_rank: int


# Sample documents for testing
SAMPLE_CANDIDATES = [
    {"id": "doc_1", "text": "TTL-based expiry removes stale cache entries automatically after a configured time period"},
    {"id": "doc_2", "text": "Cache invalidation is one of the hard problems in computer science alongside naming things"},
    {"id": "doc_3", "text": "Python decorators provide syntactic sugar for wrapping functions with additional behavior"},
    {"id": "doc_4", "text": "Write-through caching ensures data consistency by updating cache and database simultaneously"},
    {"id": "doc_5", "text": "LRU eviction policies remove least recently used items when the cache reaches capacity"},
]

SAMPLE_QUERY = "cache invalidation strategies"
