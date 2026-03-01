"""
Hybrid pipeline benchmark: sequential vs async-only vs hybrid.

Compares three approaches for a RAG-like pipeline with mixed I/O
and CPU stages. Hybrid (asyncio + ProcessPool) gives the best throughput.

From: https://dadops.dev/blog/python-concurrency-for-ai/
"""
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor

# Simulated pipeline stages
async def retrieve(query, delay=0.15):
    """I/O: fetch documents from vector DB."""
    await asyncio.sleep(delay)
    return [f"doc_{i}" for i in range(10)]

def preprocess(docs):
    """CPU: tokenize and clean documents."""
    total = 0
    for doc in docs:
        for _ in range(500_000):  # simulate CPU work
            total += 1
    return [f"processed_{d}" for d in docs]

async def generate(context, delay=0.3):
    """I/O: call LLM API with context."""
    await asyncio.sleep(delay)
    return f"Answer based on {len(context)} docs"

async def sequential_pipeline(query):
    docs = await retrieve(query)
    processed = preprocess(docs)  # blocks the event loop!
    return await generate(processed)

async def async_only_pipeline(query):
    docs = await retrieve(query)
    loop = asyncio.get_event_loop()
    processed = await loop.run_in_executor(None, preprocess, docs)
    return await generate(processed)

async def hybrid_pipeline(query, pool):
    docs = await retrieve(query)
    loop = asyncio.get_event_loop()
    processed = await loop.run_in_executor(pool, preprocess, docs)
    return await generate(processed)

async def bench(name, coro_fn, n=8, **kwargs):
    t0 = time.perf_counter()
    await asyncio.gather(*(coro_fn(f"q{i}", **kwargs)
                           for i in range(n)))
    elapsed = time.perf_counter() - t0
    print(f"{name:<20s} | {n} queries | {elapsed:.2f}s | "
          f"{n/elapsed:.1f} qps")

async def main():
    pool = ProcessPoolExecutor(max_workers=4)
    await bench("Sequential", sequential_pipeline)
    await bench("Async-only", async_only_pipeline)
    await bench("Hybrid", hybrid_pipeline, pool=pool)
    pool.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
