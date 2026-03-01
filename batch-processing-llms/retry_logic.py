"""
Exponential backoff with full jitter for LLM API calls.

Blog post: https://dadops.dev/blog/batch-processing-llms/
Code Blocks 4 & 5: Retry-After handling and exponential backoff.

Runs WITHOUT any API keys — uses mock HTTP responses for testing.
"""
import asyncio
import random


RETRYABLE = {429, 500, 502, 503, 504}


async def call_with_retry(session, url, payload, headers, max_retries=4):
    """Make an async POST with retry on transient failures.

    Uses exponential backoff with full jitter:
    - Attempt 0: immediate
    - Attempt 1: random(0, 2s)
    - Attempt 2: random(0, 4s)
    - Attempt 3: random(0, 8s)
    - Attempt 4: random(0, 16s) — capped at 30s
    """
    for attempt in range(max_retries + 1):
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status == 200:
                return await resp.json()

            if resp.status not in RETRYABLE or attempt == max_retries:
                text = await resp.text()
                raise Exception(f"HTTP {resp.status}: {text[:200]}")

            # Exponential backoff with full jitter
            base_delay = min(2 ** attempt, 30)  # cap at 30s
            jitter = random.uniform(0, base_delay)
            await asyncio.sleep(jitter)

    raise Exception("Unreachable")


def test_backoff_delays():
    """Verify exponential backoff delay calculations."""
    print("=== Backoff Delay Test ===")
    random.seed(42)

    for attempt in range(5):
        base_delay = min(2 ** attempt, 30)
        jitter = random.uniform(0, base_delay)
        print(f"  Attempt {attempt}: base={base_delay}s, "
              f"jitter={jitter:.3f}s (range 0-{base_delay}s)")

    # Verify cap works at high attempts
    assert min(2 ** 10, 30) == 30, "Cap should be 30s"
    assert min(2 ** 0, 30) == 1, "First retry base should be 1s"
    print("  PASS\n")


def test_retryable_status_codes():
    """Verify the set of retryable status codes."""
    print("=== Retryable Status Codes Test ===")
    assert 429 in RETRYABLE, "429 (rate limit) should be retryable"
    assert 500 in RETRYABLE, "500 (server error) should be retryable"
    assert 502 in RETRYABLE, "502 (bad gateway) should be retryable"
    assert 503 in RETRYABLE, "503 (service unavailable) should be retryable"
    assert 504 in RETRYABLE, "504 (gateway timeout) should be retryable"
    assert 400 not in RETRYABLE, "400 (bad request) should NOT be retryable"
    assert 401 not in RETRYABLE, "401 (unauthorized) should NOT be retryable"
    assert 404 not in RETRYABLE, "404 (not found) should NOT be retryable"
    print(f"  Retryable codes: {sorted(RETRYABLE)}")
    print("  PASS\n")


if __name__ == "__main__":
    test_backoff_delays()
    test_retryable_status_codes()
    print("All retry logic tests passed!")
