"""
Smart Batch Processor — collects requests and submits in batches at 50% discount.

From: https://dadops.dev/blog/llm-cost-optimization/
Code Block 6: "Batching and Async Processing"

Groups requests by model, auto-flushes at max batch size,
sorts by priority. Blog claims 50% batch discount.
"""

import time
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BatchRequest:
    prompt: str
    model: str
    priority: int  # 0 = low, 1 = medium, 2 = high
    callback_id: str
    input_tokens: int = 0


@dataclass
class BatchResult:
    callback_id: str
    response: str
    cost: float
    batch_discount: float


class SmartBatchProcessor:
    """Collects requests, groups by model, submits in batches."""

    BATCH_DISCOUNT = 0.50  # 50% off both input and output

    def __init__(self, flush_interval: int = 300,
                 max_batch_size: int = 100):
        self.queues: dict[str, list[BatchRequest]] = defaultdict(list)
        self.flush_interval = flush_interval
        self.max_batch_size = max_batch_size
        self.last_flush = time.time()
        self.total_saved = 0.0

    def enqueue(self, request: BatchRequest):
        """Add a request to the appropriate model queue."""
        self.queues[request.model].append(request)

        # Auto-flush if batch is full
        if len(self.queues[request.model]) >= self.max_batch_size:
            return self._flush_model(request.model)
        return []

    def flush_all(self) -> list[BatchResult]:
        """Flush all queues — call on timer or shutdown."""
        results = []
        for model in list(self.queues.keys()):
            results.extend(self._flush_model(model))
        self.last_flush = time.time()
        return results

    def _flush_model(self, model: str) -> list[BatchResult]:
        """Submit a batch for one model. Returns results."""
        requests = self.queues.pop(model, [])
        if not requests:
            return []

        # Sort by priority — high priority first in batch
        requests.sort(key=lambda r: -r.priority)

        # In production, this calls the batch API:
        # client.batches.create(requests=[...])
        results = []
        for req in requests:
            full_cost = req.input_tokens * 3.0 / 1e6  # Sonnet rate
            batch_cost = full_cost * (1 - self.BATCH_DISCOUNT)
            saved = full_cost - batch_cost
            self.total_saved += saved
            results.append(BatchResult(
                callback_id=req.callback_id,
                response=f"[batch response for {req.callback_id}]",
                cost=batch_cost,
                batch_discount=saved,
            ))
        return results

    def stats(self) -> str:
        pending = sum(len(q) for q in self.queues.values())
        return (f"Pending: {pending} requests | "
                f"Total saved: ${self.total_saved:.2f}")


if __name__ == "__main__":
    processor = SmartBatchProcessor(max_batch_size=5)

    # Simulate enqueueing requests
    print("=== Smart Batch Processor Demo ===\n")

    test_requests = [
        BatchRequest("Summarize document A", "sonnet", 1, "req-001", 3000),
        BatchRequest("Classify ticket B", "haiku", 0, "req-002", 500),
        BatchRequest("Analyze contract C", "sonnet", 2, "req-003", 8000),
        BatchRequest("Translate text D", "haiku", 0, "req-004", 1200),
        BatchRequest("Generate report E", "sonnet", 1, "req-005", 5000),
        BatchRequest("Quick lookup F", "haiku", 0, "req-006", 200),
        BatchRequest("Detailed analysis G", "sonnet", 2, "req-007", 7000),
        BatchRequest("Simple Q&A H", "haiku", 0, "req-008", 300),
    ]

    for req in test_requests:
        auto_results = processor.enqueue(req)
        if auto_results:
            print(f"  Auto-flushed {len(auto_results)} requests "
                  f"(batch full for {auto_results[0].response.split()[-1][:-1]})")
            for r in auto_results:
                print(f"    {r.callback_id}: cost=${r.cost:.6f}, "
                      f"saved=${r.batch_discount:.6f}")

    print(f"\n  {processor.stats()}")

    # Flush remaining
    remaining = processor.flush_all()
    print(f"\n  Flushed {len(remaining)} remaining requests")
    for r in remaining:
        print(f"    {r.callback_id}: cost=${r.cost:.6f}, "
              f"saved=${r.batch_discount:.6f}")

    print(f"\n  Final: {processor.stats()}")

    # Verify 50% discount math
    print(f"\n=== Discount Verification ===")
    total_full = sum(r.input_tokens * 3.0 / 1e6 for r in test_requests)
    total_batch = total_full * 0.50
    print(f"  Full cost (Sonnet input only): ${total_full:.6f}")
    print(f"  Batch cost (50% off):          ${total_batch:.6f}")
    print(f"  Savings:                       ${total_full - total_batch:.6f}")
    print(f"  Processor tracked savings:     ${processor.total_saved:.6f}")
