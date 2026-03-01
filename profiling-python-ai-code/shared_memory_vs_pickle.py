"""
Shared memory vs pickle serialization for multiprocessing.

Demonstrates the speedup from replacing implicit pickle serialization
(via multiprocessing.Pool) with shared memory for embedding results.
Uses a mock model to simulate the embedding computation.

From: https://dadops.dev/blog/profiling-python-ai-code/
"""
import time
import numpy as np
from multiprocessing import Process, Pool, shared_memory

DIM = 768  # embedding dimension

class MockModel:
    """Simulates an embedding model with deterministic output."""
    def encode(self, text):
        rng = np.random.RandomState(hash(str(text)) % 2**31)
        return rng.randn(DIM).astype(np.float32)

def load_model():
    return MockModel()

# ── BEFORE: multiprocessing.Pool with implicit pickle ──
def embed_chunk_pool(chunk):
    """Worker function for Pool.map — result gets pickled."""
    model = load_model()
    return model.encode(chunk)

def embed_with_pool(chunks, n_workers=4):
    """Old approach: Pool.map pickles every result."""
    with Pool(n_workers) as pool:
        results = pool.map(embed_chunk_pool, chunks)
    return np.array(results)

# ── AFTER: shared memory — zero-copy result collection ──
def embed_chunk_shm(chunk_indices, texts, shm_name, shape, dtype):
    """Worker writes embeddings directly to shared memory."""
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_arr = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    model = load_model()
    for idx in chunk_indices:
        shared_arr[idx] = model.encode(texts[idx])
    existing_shm.close()

def embed_with_shared_memory(chunks, n_workers=4):
    """New approach: workers write to shared numpy array."""
    n_chunks = len(chunks)
    shm = shared_memory.SharedMemory(
        create=True,
        size=n_chunks * DIM * np.dtype(np.float32).itemsize
    )
    result = np.ndarray((n_chunks, DIM), dtype=np.float32, buffer=shm.buf)

    # Split work across processes
    splits = np.array_split(range(n_chunks), n_workers)
    workers = []
    for split in splits:
        p = Process(
            target=embed_chunk_shm,
            args=(split, chunks, shm.name, (n_chunks, DIM), np.float32)
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    embeddings = result.copy()  # copy out before unlinking
    shm.close()
    shm.unlink()
    return embeddings

if __name__ == "__main__":
    # Generate test data
    chunks = [f"Document chunk {i} about machine learning topic {i % 50}"
              for i in range(500)]

    print(f"Embedding {len(chunks)} chunks with dim={DIM}")
    print(f"Data size per result: {DIM * 4} bytes (float32)")
    print()

    # Benchmark Pool (pickle-based)
    start = time.perf_counter()
    result_pool = embed_with_pool(chunks)
    pool_time = time.perf_counter() - start
    print(f"Pool (pickle):         {pool_time:.3f}s  shape={result_pool.shape}")

    # Benchmark shared memory
    start = time.perf_counter()
    result_shm = embed_with_shared_memory(chunks)
    shm_time = time.perf_counter() - start
    print(f"Shared memory:         {shm_time:.3f}s  shape={result_shm.shape}")

    # Compare
    speedup = pool_time / shm_time if shm_time > 0 else float('inf')
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Pool overhead: {pool_time - shm_time:.3f}s (pickle serialization)")
