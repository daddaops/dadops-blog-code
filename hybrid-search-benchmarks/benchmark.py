"""End-to-End Hybrid Search Benchmark

Generates a test corpus of technical documentation, runs BM25, vector, and
hybrid search across multiple query types, and measures:
- Search latency (p50, p95, p99)
- Result overlap between methods
- Basic quality indicators

Blog post: https://dadops.co/blog/hybrid-search-benchmarks/

Blog claims to verify:
- BM25 latency ~0.3ms at 10K docs
- Vector latency ~1.2ms at 10K docs
- Hybrid parallel latency ~1.5ms at 10K docs
- BM25 excels on exact keyword queries, struggles on semantic
- Vector excels on semantic queries, struggles on exact keywords
- Hybrid wins on mixed-intent queries
"""

import time
import random
import numpy as np
from bm25_search import BM25Search
from vector_search import VectorSearch
from hybrid_search import HybridSearch


def generate_corpus(n_docs: int, seed: int = 42) -> list:
    """Generate a synthetic technical documentation corpus."""
    rng = random.Random(seed)

    # Templates for realistic technical documents
    topics = [
        ("PostgreSQL", ["query optimization", "index tuning", "vacuum freezing",
                       "connection pooling", "replication setup", "EXPLAIN ANALYZE",
                       "table partitioning", "WAL configuration"]),
        ("Kubernetes", ["pod scheduling", "OOMKilled errors", "resource limits",
                       "service mesh", "ingress controller", "horizontal pod autoscaler",
                       "persistent volumes", "namespace management"]),
        ("Python", ["asyncio patterns", "memory profiling", "GIL behavior",
                   "virtual environments", "dependency management", "type hints",
                   "decorator patterns", "context managers"]),
        ("Docker", ["container networking", "multi-stage builds", "volume mounts",
                   "compose orchestration", "image optimization", "health checks",
                   "resource constraints", "logging drivers"]),
        ("Redis", ["caching strategies", "pub/sub messaging", "cluster mode",
                  "persistence options", "memory management", "Lua scripting",
                  "stream processing", "sentinel failover"]),
        ("API", ["rate limiting", "authentication", "error handling",
                "versioning", "pagination", "webhooks",
                "GraphQL vs REST", "OpenAPI specification"]),
        ("ML", ["model deployment", "feature engineering", "hyperparameter tuning",
               "A/B testing models", "data pipelines", "model monitoring",
               "batch inference", "online prediction"]),
        ("Security", ["SSL configuration", "CORS policies", "SQL injection prevention",
                     "OWASP top 10", "secret management", "audit logging",
                     "zero trust architecture", "certificate rotation"]),
    ]

    error_codes = [f"E-{rng.randint(1000, 9999)}" for _ in range(50)]
    config_params = ["setMaxPoolSize", "maxConnections", "timeout_ms",
                     "retry_count", "buffer_size", "cache_ttl",
                     "batch_size", "worker_threads", "queue_depth",
                     "max_retries", "connection_timeout", "read_timeout"]

    docs = []
    for i in range(n_docs):
        topic, subtopics = rng.choice(topics)
        subtopic = rng.choice(subtopics)
        doc_type = rng.choice(["guide", "troubleshooting", "reference", "tutorial"])

        if doc_type == "troubleshooting" and rng.random() < 0.3:
            error = rng.choice(error_codes)
            content = f"Error {error}: {subtopic} failure in {topic}. "
            content += f"This error occurs when {subtopic} is misconfigured. "
            content += f"Solution: check your {topic} {subtopic} settings and verify connectivity."
        elif doc_type == "reference" and rng.random() < 0.3:
            param = rng.choice(config_params)
            content = f"{topic} {param} parameter reference. "
            content += f"The {param} setting controls {subtopic} behavior. "
            content += f"Default value: {rng.randint(1, 1000)}. Recommended: adjust based on workload."
        else:
            content = f"{topic} {subtopic}: a comprehensive {doc_type}. "
            content += f"Learn how to implement {subtopic} in {topic} effectively. "
            content += f"Best practices for {subtopic} include proper configuration and monitoring."

        docs.append((f"doc-{i:05d}", content))

    return docs


def measure_latency(search_fn, queries, n_warmup=5, n_runs=3):
    """Measure search latency over multiple queries and runs."""
    # Warmup
    for q in queries[:n_warmup]:
        search_fn(q)

    latencies = []
    for _ in range(n_runs):
        for q in queries:
            start = time.perf_counter()
            search_fn(q)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

    latencies.sort()
    n = len(latencies)
    return {
        "p50": latencies[n // 2],
        "p95": latencies[int(n * 0.95)],
        "p99": latencies[int(n * 0.99)],
        "mean": sum(latencies) / n,
    }


def run_benchmark(n_docs: int = 1000):
    """Run the full hybrid search benchmark."""
    print(f"\n{'='*60}")
    print(f"Hybrid Search Benchmark — {n_docs} documents")
    print(f"{'='*60}")

    # Generate corpus
    print(f"\nGenerating {n_docs} synthetic documents...")
    docs = generate_corpus(n_docs)

    # Build BM25 index
    print("Building BM25 index (SQLite FTS5)...")
    t0 = time.perf_counter()
    bm25 = BM25Search()
    bm25.index(docs)
    bm25_index_time = time.perf_counter() - t0
    print(f"  BM25 index built in {bm25_index_time:.3f}s")

    # Build vector index
    print("Building vector index (FAISS + all-MiniLM-L6-v2)...")
    t0 = time.perf_counter()
    vec = VectorSearch()
    vec.index_documents(docs)
    vec_index_time = time.perf_counter() - t0
    print(f"  Vector index built in {vec_index_time:.3f}s")

    # Create hybrid
    hybrid = HybridSearch(bm25, vec)

    # Test queries by category
    keyword_queries = [
        "error E-4012 connection refused",
        "setMaxPoolSize parameter",
        "OOMKilled pod error",
        "SSL certificate expired",
        "WAL configuration PostgreSQL",
    ]

    semantic_queries = [
        "how to make queries faster",
        "preventing data loss during crashes",
        "managing memory in containers",
        "securing API endpoints",
        "automating model training pipelines",
    ]

    hybrid_intent_queries = [
        "PostgreSQL vacuum freezing best practices",
        "asyncio rate limiting patterns",
        "Kubernetes horizontal pod autoscaler tuning",
        "Redis cluster mode failover",
        "Docker multi-stage build optimization",
    ]

    all_queries = keyword_queries + semantic_queries + hybrid_intent_queries

    # Measure latency
    print("\n--- Latency Measurements ---")
    print(f"Running {len(all_queries)} queries x 3 runs each...\n")

    bm25_latency = measure_latency(
        lambda q: bm25.search(q, k=10), all_queries
    )
    vec_latency = measure_latency(
        lambda q: vec.search(q, k=10), all_queries
    )
    rrf_latency = measure_latency(
        lambda q: hybrid.search_rrf(q, k=10), all_queries
    )
    wlc_latency = measure_latency(
        lambda q: hybrid.search_wlc(q, k=10), all_queries
    )

    print(f"{'Method':<20} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'mean (ms)':>10}")
    print("-" * 62)
    for name, lat in [("BM25", bm25_latency), ("Vector", vec_latency),
                      ("Hybrid RRF", rrf_latency), ("Hybrid WLC", wlc_latency)]:
        print(f"{name:<20} {lat['p50']:>10.2f} {lat['p95']:>10.2f} {lat['p99']:>10.2f} {lat['mean']:>10.2f}")

    # Index time comparison
    print(f"\n--- Index Time ---")
    print(f"BM25:   {bm25_index_time:.3f}s")
    print(f"Vector: {vec_index_time:.3f}s")

    # Qualitative comparison
    print(f"\n--- Search Quality Comparison ---")
    print("(Showing top-3 results for representative queries)\n")

    comparison_queries = [
        ("error E-4012", "Exact keyword"),
        ("how to make queries faster", "Semantic"),
        ("PostgreSQL vacuum freezing best practices", "Hybrid-intent"),
    ]

    for query, qtype in comparison_queries:
        print(f"Query ({qtype}): '{query}'")
        bm25_res = bm25.search(query, k=3)
        vec_res = vec.search(query, k=3)
        rrf_res = hybrid.search_rrf(query, k=3)

        bm25_ids = [d for d, _ in bm25_res]
        vec_ids = [d for d, _ in vec_res]
        rrf_ids = [d for d, _ in rrf_res]

        # Show which docs appear in which method
        print(f"  BM25 top-3:   {bm25_ids}")
        print(f"  Vector top-3: {vec_ids}")
        print(f"  RRF top-3:    {rrf_ids}")

        # Check overlap
        bm25_set = set(bm25_ids)
        vec_set = set(vec_ids)
        overlap = bm25_set & vec_set
        print(f"  BM25∩Vector overlap: {len(overlap)}/3 ({overlap if overlap else 'none'})")
        print()

    # Blog claims verification
    print("=" * 60)
    print("BLOG CLAIMS VERIFICATION")
    print("=" * 60)

    blog_claims = {
        "BM25 p50 ~0.3ms (10K docs)": bm25_latency['p50'],
        "Vector p50 ~1.2ms (10K docs)": vec_latency['p50'],
        "Hybrid (seq) p50 ~1.8ms (10K docs)": rrf_latency['p50'],
    }

    for claim, actual in blog_claims.items():
        scale_note = f" (tested at {n_docs} docs)" if n_docs != 10000 else ""
        print(f"  {claim}: actual = {actual:.2f}ms{scale_note}")

    print(f"\nNote: Blog latency claims are for 10K documents.")
    print(f"This benchmark ran at {n_docs} documents.")
    if n_docs != 10000:
        print("Latency scales roughly linearly for BM25 and FAISS flat index.")


if __name__ == "__main__":
    # Start with 1K docs for quick verification
    run_benchmark(n_docs=1000)

    # If the user wants 10K, they can run:
    # run_benchmark(n_docs=10000)
