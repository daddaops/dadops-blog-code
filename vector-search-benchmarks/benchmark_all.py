"""
Full benchmark pipeline: NumPy, FAISS, and pgvector vector search.

Blog post: https://dadops.dev/blog/vector-search-benchmarks/
Code Blocks 4-8 from "Vector Search at Small Scale"

Benchmarks:
- Indexing time (how long to build each index)
- Query latency p50/p95/p99 (100 queries per config)
- Memory footprint (bytes used by each index)
- Recall@10 (accuracy vs brute-force ground truth)

Test matrix: 4 dimensions (128, 384, 768, 1536) x 3 sizes (10K, 50K, 100K)
Methods: NumPy, FAISS Flat, FAISS IVF, FAISS HNSW, pgvector HNSW, pgvector IVF

pgvector benchmarks require a running PostgreSQL instance with pgvector extension.
Set PGVECTOR_ENABLED=1 and configure PG_* env vars to enable.

Usage:
    python benchmark_all.py              # NumPy + FAISS only
    PGVECTOR_ENABLED=1 python benchmark_all.py  # Include pgvector
"""
import gc
import json
import os
import sys
import time
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("WARNING: faiss not installed. Install with: pip install faiss-cpu")

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False

PGVECTOR_ENABLED = os.environ.get("PGVECTOR_ENABLED", "0") == "1"
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DB = os.environ.get("PG_DB", "postgres")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASS = os.environ.get("PG_PASS", "secret")

# Test matrix
DIMS = [128, 384, 768, 1536]
SIZES = [10_000, 50_000, 100_000]
N_QUERIES = 100
K = 10


def log(msg=""):
    print(msg, flush=True)


def generate_data(n_vectors, dim):
    """Generate random unit vectors (cosine sim = dot product)."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n_vectors, dim)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def generate_queries(n_queries, dim):
    """Separate RNG seed for query vectors."""
    rng = np.random.default_rng(123)
    vecs = rng.standard_normal((n_queries, dim)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def measure_latency(search_fn, queries, warmup=10):
    """Run search_fn for each query, return latency array in ms."""
    # Warmup
    for i in range(min(warmup, len(queries))):
        search_fn(queries[i])

    times = []
    for q in queries:
        t0 = time.perf_counter()
        search_fn(q)
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def compute_recall(results, ground_truth, k=10):
    """Compute recall@k: fraction of ground truth neighbors found."""
    recalls = []
    for r, gt in zip(results, ground_truth):
        r_set = set(r[:k])
        gt_set = set(gt[:k])
        recalls.append(len(r_set & gt_set) / k)
    return np.mean(recalls)


# ──────────────────────────────────────────────
# NumPy brute-force search
# ──────────────────────────────────────────────

def numpy_search(query, database, k=10):
    scores = query @ database.T
    top_k = np.argpartition(-scores, k)[:k]
    return top_k[np.argsort(-scores[top_k])]


def bench_numpy(database, queries, k=10):
    """Benchmark NumPy brute-force search."""
    # "Indexing" time (just loading into memory — trivially fast)
    t0 = time.perf_counter()
    _ = database.copy()  # simulate "loading"
    index_time = time.perf_counter() - t0

    # Query latency
    latencies = measure_latency(
        lambda q: numpy_search(q, database, k),
        queries
    )

    # Ground truth (numpy IS the ground truth)
    gt_results = [numpy_search(q, database, k) for q in queries]

    # Memory: just the raw array
    mem_bytes = database.nbytes

    return {
        "method": "NumPy",
        "index_time_s": index_time,
        "latency_ms": latencies,
        "mem_bytes": mem_bytes,
        "results": gt_results,
        "recall": 1.0,  # exact by definition
    }


# ──────────────────────────────────────────────
# FAISS benchmarks
# ──────────────────────────────────────────────

def bench_faiss_flat(database, queries, k=10):
    """Benchmark FAISS IndexFlatIP (exact brute force)."""
    if not HAS_FAISS:
        return None

    d = database.shape[1]
    db = database.copy()

    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(d)
    index.add(db)
    index_time = time.perf_counter() - t0

    def search_fn(q):
        D, I = index.search(q.reshape(1, -1), k)
        return I[0]

    latencies = measure_latency(search_fn, queries)
    results = [search_fn(q) for q in queries]

    return {
        "method": "FAISS Flat",
        "index_time_s": index_time,
        "latency_ms": latencies,
        "mem_bytes": database.nbytes,  # Flat stores raw data only
        "results": results,
        "recall": None,  # compute after ground truth established
    }


def bench_faiss_ivf(database, queries, k=10, nlist=100, nprobe=10):
    """Benchmark FAISS IndexIVFFlat."""
    if not HAS_FAISS:
        return None

    d = database.shape[1]
    db = database.copy()

    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(db)
    index.add(db)
    index.nprobe = nprobe
    index_time = time.perf_counter() - t0

    def search_fn(q):
        D, I = index.search(q.reshape(1, -1), k)
        return I[0]

    latencies = measure_latency(search_fn, queries)
    results = [search_fn(q) for q in queries]

    return {
        "method": "FAISS IVF",
        "index_time_s": index_time,
        "latency_ms": latencies,
        "mem_bytes": database.nbytes + nlist * d * 4,  # data + centroids
        "results": results,
        "recall": None,
    }


def bench_faiss_hnsw(database, queries, k=10, M=32, ef_construction=64, ef_search=40):
    """Benchmark FAISS IndexHNSWFlat."""
    if not HAS_FAISS:
        return None

    d = database.shape[1]
    db = database.copy()

    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(db)
    index.hnsw.efSearch = ef_search
    index_time = time.perf_counter() - t0

    def search_fn(q):
        D, I = index.search(q.reshape(1, -1), k)
        return I[0]

    latencies = measure_latency(search_fn, queries)
    results = [search_fn(q) for q in queries]

    return {
        "method": "FAISS HNSW",
        "index_time_s": index_time,
        "latency_ms": latencies,
        "mem_bytes": None,  # hard to measure HNSW graph size externally
        "results": results,
        "recall": None,
    }


# ──────────────────────────────────────────────
# pgvector benchmarks
# ──────────────────────────────────────────────

def get_pg_conn():
    """Get a PostgreSQL connection with pgvector registered."""
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASS
    )
    register_vector(conn)
    conn.autocommit = True
    return conn


def bench_pgvector_hnsw(database, queries, k=10, m=16, ef_construction=64, ef_search=40):
    """Benchmark pgvector with HNSW index."""
    if not HAS_PGVECTOR or not PGVECTOR_ENABLED:
        return None

    d = database.shape[1]
    n = database.shape[0]
    conn = get_pg_conn()
    cur = conn.cursor()

    # Setup
    cur.execute("DROP TABLE IF EXISTS bench_items")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"CREATE TABLE bench_items (id serial PRIMARY KEY, embedding vector({d}))")

    # Insert vectors
    t0 = time.perf_counter()
    for i in range(n):
        cur.execute(
            "INSERT INTO bench_items (embedding) VALUES (%s)",
            (database[i],)
        )

    # Create HNSW index
    cur.execute(f"""
        CREATE INDEX ON bench_items
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = {m}, ef_construction = {ef_construction})
    """)
    index_time = time.perf_counter() - t0

    # Set search params
    cur.execute(f"SET hnsw.ef_search = {ef_search}")

    def search_fn(q):
        cur.execute(
            "SELECT id FROM bench_items ORDER BY embedding <=> %s LIMIT %s",
            (q, k)
        )
        return np.array([row[0] - 1 for row in cur.fetchall()])  # 0-indexed

    latencies = measure_latency(search_fn, queries)
    results = [search_fn(q) for q in queries]

    # Memory (approximate)
    cur.execute("SELECT pg_total_relation_size('bench_items')")
    mem_bytes = cur.fetchone()[0]

    cur.execute("DROP TABLE bench_items")
    conn.close()

    return {
        "method": "pgvector HNSW",
        "index_time_s": index_time,
        "latency_ms": latencies,
        "mem_bytes": mem_bytes,
        "results": results,
        "recall": None,
    }


def bench_pgvector_ivf(database, queries, k=10, lists=100, probes=10):
    """Benchmark pgvector with IVFFlat index."""
    if not HAS_PGVECTOR or not PGVECTOR_ENABLED:
        return None

    d = database.shape[1]
    n = database.shape[0]
    conn = get_pg_conn()
    cur = conn.cursor()

    # Setup
    cur.execute("DROP TABLE IF EXISTS bench_items_ivf")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"CREATE TABLE bench_items_ivf (id serial PRIMARY KEY, embedding vector({d}))")

    # Insert vectors
    t0 = time.perf_counter()
    for i in range(n):
        cur.execute(
            "INSERT INTO bench_items_ivf (embedding) VALUES (%s)",
            (database[i],)
        )

    # Create IVF index
    cur.execute(f"""
        CREATE INDEX ON bench_items_ivf
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {lists})
    """)
    index_time = time.perf_counter() - t0

    # Set search params
    cur.execute(f"SET ivfflat.probes = {probes}")

    def search_fn(q):
        cur.execute(
            "SELECT id FROM bench_items_ivf ORDER BY embedding <=> %s LIMIT %s",
            (q, k)
        )
        return np.array([row[0] - 1 for row in cur.fetchall()])  # 0-indexed

    latencies = measure_latency(search_fn, queries)
    results = [search_fn(q) for q in queries]

    # Memory
    cur.execute("SELECT pg_total_relation_size('bench_items_ivf')")
    mem_bytes = cur.fetchone()[0]

    cur.execute("DROP TABLE bench_items_ivf")
    conn.close()

    return {
        "method": "pgvector IVF",
        "index_time_s": index_time,
        "latency_ms": latencies,
        "mem_bytes": mem_bytes,
        "results": results,
        "recall": None,
    }


# ──────────────────────────────────────────────
# Main benchmark runner
# ──────────────────────────────────────────────

def run_benchmark(dim, n_vectors):
    """Run all benchmarks for a given dimension and dataset size."""
    log(f"\n{'='*60}")
    log(f"BENCHMARK: {n_vectors:,} vectors, {dim} dimensions")
    log(f"{'='*60}")

    database = generate_data(n_vectors, dim)
    queries = generate_queries(N_QUERIES, dim)
    log(f"Generated {n_vectors:,} database vectors and {N_QUERIES} queries")

    results = {}

    # NumPy
    log("\n  NumPy brute-force...")
    r = bench_numpy(database, queries, K)
    results["numpy"] = r
    gt = r["results"]  # ground truth
    log(f"    Index: {r['index_time_s']:.4f}s | p50: {np.percentile(r['latency_ms'], 50):.3f}ms | "
        f"Mem: {r['mem_bytes']/1e6:.1f}MB")

    # FAISS Flat
    if HAS_FAISS:
        log("  FAISS Flat...")
        r = bench_faiss_flat(database, queries, K)
        r["recall"] = compute_recall(r["results"], gt, K)
        results["faiss_flat"] = r
        log(f"    Index: {r['index_time_s']:.4f}s | p50: {np.percentile(r['latency_ms'], 50):.3f}ms | "
            f"Recall@{K}: {r['recall']:.2%} | Mem: {r['mem_bytes']/1e6:.1f}MB")

        # FAISS IVF
        log("  FAISS IVF...")
        r = bench_faiss_ivf(database, queries, K)
        r["recall"] = compute_recall(r["results"], gt, K)
        results["faiss_ivf"] = r
        log(f"    Index: {r['index_time_s']:.4f}s | p50: {np.percentile(r['latency_ms'], 50):.3f}ms | "
            f"Recall@{K}: {r['recall']:.2%} | Mem: {r['mem_bytes']/1e6:.1f}MB")

        # FAISS HNSW
        log("  FAISS HNSW...")
        r = bench_faiss_hnsw(database, queries, K)
        r["recall"] = compute_recall(r["results"], gt, K)
        results["faiss_hnsw"] = r
        log(f"    Index: {r['index_time_s']:.4f}s | p50: {np.percentile(r['latency_ms'], 50):.3f}ms | "
            f"Recall@{K}: {r['recall']:.2%}")

    # pgvector
    if PGVECTOR_ENABLED and HAS_PGVECTOR:
        log("  pgvector HNSW...")
        r = bench_pgvector_hnsw(database, queries, K)
        if r:
            r["recall"] = compute_recall(r["results"], gt, K)
            results["pgvector_hnsw"] = r
            log(f"    Index: {r['index_time_s']:.1f}s | p50: {np.percentile(r['latency_ms'], 50):.3f}ms | "
                f"Recall@{K}: {r['recall']:.2%} | Mem: {r['mem_bytes']/1e6:.1f}MB")

        log("  pgvector IVF...")
        r = bench_pgvector_ivf(database, queries, K)
        if r:
            r["recall"] = compute_recall(r["results"], gt, K)
            results["pgvector_ivf"] = r
            log(f"    Index: {r['index_time_s']:.1f}s | p50: {np.percentile(r['latency_ms'], 50):.3f}ms | "
                f"Recall@{K}: {r['recall']:.2%} | Mem: {r['mem_bytes']/1e6:.1f}MB")
    elif not PGVECTOR_ENABLED:
        log("  pgvector: SKIPPED (set PGVECTOR_ENABLED=1 to enable)")

    gc.collect()
    return results


def main():
    log("Vector Search Benchmark")
    log(f"Test matrix: {DIMS} dims x {SIZES} sizes")
    log(f"Queries: {N_QUERIES}, k={K}")
    log(f"FAISS available: {HAS_FAISS}")
    log(f"pgvector enabled: {PGVECTOR_ENABLED} (available: {HAS_PGVECTOR})")

    all_results = {}

    for dim in DIMS:
        for n in SIZES:
            key = f"{dim}d_{n//1000}K"
            all_results[key] = run_benchmark(dim, n)

    # Save results summary
    summary = {}
    for key, methods in all_results.items():
        summary[key] = {}
        for mname, r in methods.items():
            summary[key][mname] = {
                "method": r["method"],
                "index_time_s": round(r["index_time_s"], 4),
                "p50_ms": round(float(np.percentile(r["latency_ms"], 50)), 3),
                "p95_ms": round(float(np.percentile(r["latency_ms"], 95)), 3),
                "p99_ms": round(float(np.percentile(r["latency_ms"], 99)), 3),
                "recall": round(r["recall"], 4) if r["recall"] is not None else None,
                "mem_mb": round(r["mem_bytes"] / 1e6, 1) if r["mem_bytes"] is not None else None,
            }

    with open("output/benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nResults saved to output/benchmark_results.json")

    # Print summary tables
    log("\n" + "=" * 80)
    log("SUMMARY TABLES")
    log("=" * 80)

    for dim in DIMS:
        log(f"\n--- {dim} dimensions ---")
        log(f"{'Method':<16} {'Size':>8} {'Index(s)':>10} {'p50(ms)':>10} {'p95(ms)':>10} "
            f"{'p99(ms)':>10} {'Recall@10':>10} {'Mem(MB)':>10}")
        log("-" * 86)
        for n in SIZES:
            key = f"{dim}d_{n//1000}K"
            if key not in all_results:
                continue
            for mname in ["numpy", "faiss_flat", "faiss_ivf", "faiss_hnsw",
                          "pgvector_hnsw", "pgvector_ivf"]:
                if mname not in all_results[key]:
                    continue
                r = all_results[key][mname]
                lat = r["latency_ms"]
                recall_str = f"{r['recall']:.2%}" if r["recall"] is not None else "N/A"
                mem_str = f"{r['mem_bytes']/1e6:.1f}" if r["mem_bytes"] is not None else "N/A"
                log(f"{r['method']:<16} {n//1000:>6}K {r['index_time_s']:>10.4f} "
                    f"{np.percentile(lat, 50):>10.3f} {np.percentile(lat, 95):>10.3f} "
                    f"{np.percentile(lat, 99):>10.3f} {recall_str:>10} {mem_str:>10}")


if __name__ == "__main__":
    main()
