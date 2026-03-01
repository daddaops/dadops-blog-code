"""
Production async patterns: retry with backoff, semaphore, TaskGroup.

Demonstrates structured concurrency with automatic cleanup,
rate limiting via semaphore, and exponential backoff retry logic.

From: https://dadops.dev/blog/python-concurrency-for-ai/
"""
import asyncio
import signal
import random

async def call_api_with_retry(session_id, semaphore,
                               max_retries=3, base_delay=1.0):
    """Production-ready async API call with rate limiting
    and exponential backoff."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                # Simulated API call
                await asyncio.sleep(0.1 + random.random() * 0.2)
                if random.random() < 0.1:  # 10% failure rate
                    raise ConnectionError("API timeout")
                return {"session": session_id, "status": "ok"}
            except (ConnectionError, TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

async def process_batch(requests, max_concurrent=50):
    """Structured concurrency: all tasks are cleaned up
    even if one fails."""
    semaphore = asyncio.Semaphore(max_concurrent)
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(
                call_api_with_retry(req_id, semaphore)
            )
            for req_id in requests
        ]
    return [t.result() for t in tasks]

async def main():
    # Graceful shutdown on SIGTERM
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM,
                            shutdown_event.set)

    requests = list(range(200))
    try:
        results = await process_batch(requests)
        print(f"Completed {len(results)} requests")
    except ExceptionGroup as eg:
        failed = len(eg.exceptions)
        print(f"{failed} requests failed after retries")

if __name__ == "__main__":
    asyncio.run(main())
