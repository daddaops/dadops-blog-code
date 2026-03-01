"""
Complete BatchProcessor class for production LLM batch processing.

Blog post: https://dadops.dev/blog/batch-processing-llms/
Code Block 9: Full BatchProcessor combining all utilities.

REQUIRES: OpenAI API key (set OPENAI_API_KEY environment variable)
Without an API key, this module can be imported but not run.
"""
import asyncio
import aiohttp
import hashlib
import json
import random
import time
from pathlib import Path


class BatchProcessor:
    """Production-grade LLM batch processor.

    Combines: async concurrency, rate limiting, retry with jitter,
    SHA-256 prompt cache, JSONL checkpointing, cost tracking with
    budget enforcement.
    """
    RETRYABLE = {429, 500, 502, 503, 504}
    PRICING = {
        "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00),
        "claude-haiku": (1.00, 5.00), "claude-sonnet": (3.00, 15.00),
    }

    def __init__(self, model="gpt-4o-mini", api_key="", max_concurrent=10,
                 max_retries=4, budget=None, checkpoint_path=None):
        self.model = model
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.budget = budget
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.cache = {}
        self.cache_hits = 0
        # Checkpoint
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.completed = {}
        if self.checkpoint_path and self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                for line in f:
                    rec = json.loads(line)
                    self.completed[rec["id"]] = rec["result"]

    @property
    def cost(self):
        inp, out = self.PRICING.get(self.model, (0, 0))
        return self.input_tokens / 1e6 * inp + self.output_tokens / 1e6 * out

    def _cache_key(self, messages):
        raw = json.dumps({"m": self.model, "msg": messages}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _call(self, session, semaphore, rate_bucket, messages):
        # Check cache first
        ck = self._cache_key(messages)
        if ck in self.cache:
            self.cache_hits += 1
            return self.cache[ck]

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages}

        async with semaphore:
            await rate_bucket.acquire()
            for attempt in range(self.max_retries + 1):
                try:
                    async with session.post(url, json=payload,
                                            headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=60)
                                            ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            usage = data["usage"]
                            self.input_tokens += usage["prompt_tokens"]
                            self.output_tokens += usage["completion_tokens"]
                            self.calls += 1
                            result = data["choices"][0]["message"]["content"]
                            self.cache[ck] = result
                            return result

                        if resp.status not in self.RETRYABLE \
                                or attempt == self.max_retries:
                            text = await resp.text()
                            return f"ERROR {resp.status}: {text[:200]}"

                        delay = min(2 ** attempt, 30)
                        await asyncio.sleep(random.uniform(0, delay))
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt == self.max_retries:
                        return "ERROR: connection failed"
                    await asyncio.sleep(random.uniform(0, 2 ** attempt))

    async def run(self, items, prompt_fn, id_fn=None):
        """Process items through the LLM.

        Args:
            items:     list of objects to process
            prompt_fn: item -> list of messages
            id_fn:     item -> unique string ID (for checkpointing)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        class _RateBucket:
            def __init__(self, rate):
                self.rate, self.tokens = rate, rate
                self.last = time.monotonic()
            async def acquire(self):
                while True:
                    now = time.monotonic()
                    self.tokens = min(self.rate,
                        self.tokens + (now - self.last) * self.rate / 60)
                    self.last = now
                    if self.tokens >= 1:
                        self.tokens -= 1; return
                    await asyncio.sleep(0.05)

        rate_bucket = _RateBucket(self.max_concurrent * 2)
        results = [None] * len(items)
        total = len(items)

        async with aiohttp.ClientSession() as session:
            async def process(i, item):
                item_id = id_fn(item) if id_fn else str(i)
                # Skip if checkpointed
                if item_id in self.completed:
                    results[i] = self.completed[item_id]
                    return
                if self.budget and self.cost >= self.budget:
                    results[i] = "BUDGET_EXCEEDED"
                    return
                messages = prompt_fn(item)
                result = await self._call(session, semaphore,
                                          rate_bucket, messages)
                results[i] = result
                # Checkpoint
                if self.checkpoint_path and result \
                        and not result.startswith("ERROR"):
                    self.completed[item_id] = result
                    with open(self.checkpoint_path, "a") as f:
                        f.write(json.dumps({"id": item_id,
                                            "result": result}) + "\n")
                        f.flush()
                # Progress
                done = sum(1 for r in results if r is not None)
                if done % 100 == 0 or done == total:
                    projected = self.cost / max(self.calls, 1) * total
                    print(f"[{done}/{total}] ${self.cost:.4f} spent"
                          f" | ~${projected:.2f} projected"
                          f" | cache hits: {self.cache_hits}")

            await asyncio.gather(*[process(i, item)
                                   for i, item in enumerate(items)])
        return results


if __name__ == "__main__":
    print("BatchProcessor class loaded successfully.")
    print("To run, set OPENAI_API_KEY and use example_usage.py")
    print(f"Supported models: {list(BatchProcessor.PRICING.keys())}")
