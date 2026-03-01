"""
Standalone utility classes for LLM batch processing.

Blog post: https://dadops.dev/blog/batch-processing-llms/
Code Blocks 3, 6, 7, 8: TokenBucket, PromptCache, CostTracker, Checkpoint.

These classes run WITHOUT any API keys — they are pure utility code.
"""
import asyncio
import hashlib
import json
import time
from pathlib import Path


# ── Code Block 3: Token Bucket Rate Limiter ──

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate            # tokens added per second
        self.capacity = capacity    # max burst size
        self.tokens = capacity
        self.last_refill = time.monotonic()

    async def acquire(self):
        while True:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return
            # Wait just long enough for one token to appear
            await asyncio.sleep((1 - self.tokens) / self.rate)


# ── Code Block 6: Hash-Based Prompt Cache ──

class PromptCache:
    def __init__(self):
        self.store = {}
        self.hits = 0
        self.misses = 0

    def _key(self, model, messages, temperature=1.0):
        raw = json.dumps({"m": model, "msg": messages, "t": temperature},
                         sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model, messages, temperature=1.0):
        key = self._key(model, messages, temperature)
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        return None

    def put(self, model, messages, temperature, result):
        key = self._key(model, messages, temperature)
        self.store[key] = result

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


# ── Code Block 7: Cost Tracker ──

class CostTracker:
    PRICING = {  # per million tokens: (input, output)
        "gpt-4o-mini":    (0.15,  0.60),
        "gpt-4o":         (2.50,  10.00),
        "claude-haiku":   (1.00,  5.00),
        "claude-sonnet":  (3.00,  15.00),
    }

    def __init__(self, model, budget=None):
        self.model = model
        self.budget = budget
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def record(self, usage):
        self.input_tokens += usage["prompt_tokens"]
        self.output_tokens += usage["completion_tokens"]
        self.calls += 1

    @property
    def total_cost(self):
        inp, out = self.PRICING[self.model]
        return (self.input_tokens / 1_000_000 * inp +
                self.output_tokens / 1_000_000 * out)

    @property
    def over_budget(self):
        return self.budget is not None and self.total_cost >= self.budget

    def summary(self, total_items):
        projected = self.total_cost / self.calls * total_items if self.calls else 0
        return (f"[{self.calls}/{total_items}] "
                f"${self.total_cost:.4f} spent | "
                f"~${projected:.2f} projected total")


# ── Code Block 8: JSONL Checkpoint ──

class Checkpoint:
    def __init__(self, path="batch_checkpoint.jsonl"):
        self.path = Path(path)
        self.completed = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    record = json.loads(line)
                    self.completed[record["id"]] = record["result"]
            print(f"Resumed: {len(self.completed)} items from checkpoint")

    def is_done(self, item_id):
        return item_id in self.completed

    def get(self, item_id):
        return self.completed.get(item_id)

    def save(self, item_id, result):
        self.completed[item_id] = result
        with open(self.path, "a") as f:
            f.write(json.dumps({"id": item_id, "result": result}) + "\n")
            f.flush()


# ── Self-tests (no API keys needed) ──

def test_token_bucket():
    """Verify TokenBucket allows burst then throttles."""
    async def _test():
        bucket = TokenBucket(rate=500/60, capacity=20)  # 500 RPM, burst 20
        # Should allow 20 immediate acquires (burst capacity)
        t0 = time.monotonic()
        for _ in range(20):
            await bucket.acquire()
        burst_time = time.monotonic() - t0
        print(f"  TokenBucket burst of 20: {burst_time*1000:.1f}ms (should be near 0)")
        assert burst_time < 0.1, f"Burst took too long: {burst_time:.3f}s"

        # Next acquire should wait ~120ms (1 token / 8.33 tokens/sec)
        t0 = time.monotonic()
        await bucket.acquire()
        wait_time = time.monotonic() - t0
        print(f"  TokenBucket 21st acquire wait: {wait_time*1000:.1f}ms (should be ~120ms)")
        assert wait_time > 0.05, f"Should have waited but got {wait_time:.3f}s"

    print("=== TokenBucket Test ===")
    asyncio.run(_test())
    print("  PASS\n")


def test_prompt_cache():
    """Verify PromptCache detects duplicates."""
    print("=== PromptCache Test ===")
    cache = PromptCache()
    messages = [{"role": "user", "content": "Hello"}]

    # First call: miss
    result = cache.get("gpt-4o-mini", messages)
    assert result is None
    assert cache.misses == 1

    # Store result
    cache.put("gpt-4o-mini", messages, 1.0, "Hi there!")

    # Second call: hit
    result = cache.get("gpt-4o-mini", messages)
    assert result == "Hi there!"
    assert cache.hits == 1
    assert cache.hit_rate == 0.5  # 1 hit, 1 miss

    # Different temperature: miss
    result = cache.get("gpt-4o-mini", messages, temperature=0.5)
    assert result is None
    assert cache.misses == 2

    print(f"  Hit rate: {cache.hit_rate:.0%} (1 hit, 2 misses)")
    print("  PASS\n")


def test_cost_tracker():
    """Verify CostTracker calculates costs correctly."""
    print("=== CostTracker Test ===")
    tracker = CostTracker("gpt-4o-mini", budget=1.00)

    # Simulate 1000 calls, ~100 input tokens + ~20 output tokens each
    for _ in range(1000):
        tracker.record({"prompt_tokens": 100, "completion_tokens": 20})

    # Expected: 100K input tokens * $0.15/M + 20K output tokens * $0.60/M
    # = $0.015 + $0.012 = $0.027
    expected = 100_000 / 1_000_000 * 0.15 + 20_000 / 1_000_000 * 0.60
    print(f"  1000 calls: ${tracker.total_cost:.4f} (expected ${expected:.4f})")
    assert abs(tracker.total_cost - expected) < 0.0001

    print(f"  Summary: {tracker.summary(5000)}")
    print(f"  Over budget ($1.00)? {tracker.over_budget}")
    assert not tracker.over_budget
    print("  PASS\n")


def test_checkpoint():
    """Verify Checkpoint save/resume cycle."""
    import tempfile, os
    print("=== Checkpoint Test ===")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                      delete=False) as f:
        tmp_path = f.name

    try:
        # Write some checkpoints
        cp = Checkpoint(tmp_path)
        cp.save("item_1", "category_a")
        cp.save("item_2", "category_b")
        cp.save("item_3", "category_c")
        assert len(cp.completed) == 3

        # Simulate crash and resume
        cp2 = Checkpoint(tmp_path)
        assert len(cp2.completed) == 3
        assert cp2.get("item_1") == "category_a"
        assert cp2.get("item_3") == "category_c"
        assert cp2.is_done("item_2")
        assert not cp2.is_done("item_999")

        print(f"  Saved 3 items, resumed 3 items")
        print("  PASS\n")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_token_bucket()
    test_prompt_cache()
    test_cost_tracker()
    test_checkpoint()
    print("All utility tests passed!")
