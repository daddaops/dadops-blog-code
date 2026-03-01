"""
asyncio vs threading for concurrent API calls.

Compares async (coroutines with semaphore) vs threading (ThreadPoolExecutor)
for batch API call patterns at various scales.

From: https://dadops.dev/blog/python-concurrency-for-ai/
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- asyncio approach ---
async def async_api_call(task_id, delay=0.3):
    """Simulated async API call (e.g., AsyncOpenAI)."""
    await asyncio.sleep(delay)
    return {"id": task_id, "tokens": 42}

async def run_async_batch(n_requests, max_concurrent=20):
    semaphore = asyncio.Semaphore(max_concurrent)
    async def limited(task_id):
        async with semaphore:
            return await async_api_call(task_id)
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(limited(i)) for i in range(n_requests)]
    return [t.result() for t in tasks]

# --- threading approach ---
def sync_api_call(task_id, delay=0.3):
    """Simulated sync API call (e.g., OpenAI)."""
    time.sleep(delay)
    return {"id": task_id, "tokens": 42}

def run_threaded_batch(n_requests, max_workers=20):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(sync_api_call, i): i
                   for i in range(n_requests)}
        for f in as_completed(futures):
            results.append(f.result())
    return results

# --- benchmark ---
if __name__ == "__main__":
    for n in [10, 50, 100, 500]:
        t0 = time.perf_counter()
        asyncio.run(run_async_batch(n))
        t_async = time.perf_counter() - t0

        t0 = time.perf_counter()
        run_threaded_batch(n)
        t_thread = time.perf_counter() - t0

        print(f"n={n:>3d} | asyncio: {t_async:.2f}s | "
              f"threading: {t_thread:.2f}s | "
              f"ratio: {t_thread/t_async:.2f}x")
