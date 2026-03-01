"""
Memory profiling with tracemalloc: detecting memory leaks.

Demonstrates the common "accumulate-then-process" memory pattern
and its fix using np.memmap for streaming results to disk.
Uses a mock model to simulate embedding generation.

From: https://dadops.dev/blog/profiling-python-ai-code/
"""
import tracemalloc
import numpy as np
import tempfile
import os

DIM = 768  # embedding dimension

class MockModel:
    """Simulates an embedding model with deterministic output."""
    def encode(self, texts, batch_size=None):
        """Encode a batch of texts into embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts) if batch_size is None else min(len(texts), batch_size)
        rng = np.random.RandomState(42)
        return rng.randn(n, DIM).astype(np.float32)

def process_batch(texts, model, batch_size=64):
    """Generate embeddings for a batch of texts."""
    return model.encode(texts[:batch_size], batch_size=batch_size)

def run_embedding_job_leaky(all_texts, model, batch_size=64):
    """BUG: accumulates all results in memory."""
    all_results = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]
        embeddings = process_batch(batch, model, batch_size)
        all_results.append(embeddings)  # memory grows forever!
    return np.vstack(all_results)       # also doubles memory here

def run_embedding_job_fixed(all_texts, model, output_path, batch_size=64):
    """Stream results to a memory-mapped file."""
    n = len(all_texts)
    # Pre-allocate output file
    fp = np.memmap(output_path, dtype=np.float32,
                   mode="w+", shape=(n, DIM))
    for i in range(0, n, batch_size):
        batch = all_texts[i:i + batch_size]
        embeddings = process_batch(batch, model, batch_size)
        fp[i:i + len(embeddings)] = embeddings
        fp.flush()  # write to disk, free memory
    del fp

if __name__ == "__main__":
    model = MockModel()
    n_docs = 10000
    texts = [f"Document {i} about topic {i % 100}" for i in range(n_docs)]

    print(f"Processing {n_docs} documents (dim={DIM})")
    print(f"Expected memory per batch: {64 * DIM * 4 / 1024:.1f} KB")
    print(f"Expected total for all results: {n_docs * DIM * 4 / 1024 / 1024:.1f} MB")
    print()

    # ── Leaky version with tracemalloc ──
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    result_leaky = run_embedding_job_leaky(texts, model)

    snapshot2 = tracemalloc.take_snapshot()

    print("=== LEAKY VERSION: tracemalloc top allocations ===")
    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    for stat in top_stats[:5]:
        print(f"  {stat}")
    print(f"\nLeaky result shape: {result_leaky.shape}")
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory:    {peak / 1024 / 1024:.1f} MB")

    tracemalloc.stop()
    del result_leaky

    # ── Fixed version with memmap ──
    print("\n=== FIXED VERSION: streaming to memmap ===")
    tracemalloc.start()
    snapshot3 = tracemalloc.take_snapshot()

    tmp_path = tempfile.mktemp(suffix=".dat")
    run_embedding_job_fixed(texts, model, tmp_path)

    snapshot4 = tracemalloc.take_snapshot()
    top_stats_fixed = snapshot4.compare_to(snapshot3, "lineno")
    for stat in top_stats_fixed[:5]:
        print(f"  {stat}")

    current_fixed, peak_fixed = tracemalloc.get_traced_memory()
    print(f"\nFixed current memory: {current_fixed / 1024 / 1024:.1f} MB")
    print(f"Fixed peak memory:    {peak_fixed / 1024 / 1024:.1f} MB")

    # Verify output
    result_fixed = np.memmap(tmp_path, dtype=np.float32,
                             mode="r", shape=(n_docs, DIM))
    print(f"Fixed result shape: {result_fixed.shape}")

    tracemalloc.stop()
    os.unlink(tmp_path)

    print(f"\nMemory reduction: {peak / peak_fixed:.1f}x less peak memory")
