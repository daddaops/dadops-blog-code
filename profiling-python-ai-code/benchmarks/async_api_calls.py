"""Async I/O workload: simulated concurrent API calls."""
import asyncio

async def fake_api_call(i):
    """Simulate an API call with variable latency."""
    await asyncio.sleep(0.01)  # 10ms simulated latency
    return {"id": i, "result": f"response_{i}"}

async def main():
    tasks = [fake_api_call(i) for i in range(50)]
    results = await asyncio.gather(*tasks)
    return results

if __name__ == "__main__":
    asyncio.run(main())
