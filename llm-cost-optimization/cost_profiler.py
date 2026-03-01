"""
LLM Cost Profiler — tracks per-feature cost breakdown.

From: https://dadops.dev/blog/llm-cost-optimization/
Code Block 1: "Understanding Your LLM Cost Structure"

Wraps LLM API calls to track input/output tokens and costs per feature.
Generates a report showing which features consume the most spend.
"""

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CostRecord:
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0
    total_cost: float = 0.0


class LLMCostProfiler:
    """Wraps LLM calls to track per-feature cost breakdown."""

    PRICING = {  # per million tokens, Feb 2026
        "opus":   {"input": 15.00, "output": 75.00},
        "sonnet": {"input":  3.00, "output": 15.00},
        "haiku":  {"input":  1.00, "output":  5.00},
    }

    def __init__(self):
        self.records = defaultdict(CostRecord)

    def track(self, feature: str, model: str,
              input_tokens: int, output_tokens: int):
        pricing = self.PRICING[model]
        cost = (input_tokens * pricing["input"]
                + output_tokens * pricing["output"]) / 1_000_000
        rec = self.records[feature]
        rec.input_tokens += input_tokens
        rec.output_tokens += output_tokens
        rec.request_count += 1
        rec.total_cost += cost

    def report(self) -> str:
        total = sum(r.total_cost for r in self.records.values())
        lines = ["=== LLM Cost Report ==="]
        for feat, rec in sorted(self.records.items(),
                                key=lambda x: -x[1].total_cost):
            pct = (rec.total_cost / total * 100) if total else 0
            input_pct = (rec.input_tokens * 100
                         / (rec.input_tokens + rec.output_tokens))
            lines.append(
                f"  {feat}: ${rec.total_cost:.2f}/day "
                f"({pct:.0f}% of total) | "
                f"{input_pct:.0f}% input tokens | "
                f"{rec.request_count} requests"
            )
        lines.append(f"  TOTAL: ${total:.2f}/day "
                     f"(${total * 30:.0f}/month)")
        return "\n".join(lines)


if __name__ == "__main__":
    profiler = LLMCostProfiler()

    # Simulate the blog's three workload profiles for one day:

    # Chatbot: 10K conversations/day, 4,200 input + 350 output tokens (Sonnet)
    # Blog claims: $5,355/month, 71% input-driven
    for _ in range(10_000):
        profiler.track("chatbot", "sonnet", 4200, 350)

    # Document processor: 1K PDFs/day, 12,400 input + 340 output tokens (Sonnet)
    # Blog claims: $1,269/month, 88% input-driven
    for _ in range(1_000):
        profiler.track("document-processor", "sonnet", 12400, 340)

    # Code generator: 5K requests/day, 800 input + 1,200 output tokens (Sonnet)
    # Blog claims: $3,060/month, 88% output-driven
    for _ in range(5_000):
        profiler.track("code-generator", "sonnet", 800, 1200)

    print(profiler.report())

    # Verify blog claims
    print("\n=== Blog Claim Verification ===")
    for feat, rec in profiler.records.items():
        monthly = rec.total_cost * 30
        input_pct = rec.input_tokens * 100 / (rec.input_tokens + rec.output_tokens)
        print(f"  {feat}: ${monthly:.0f}/month, {input_pct:.0f}% input tokens")

    # Manual math verification:
    # chatbot: (4200*3.0 + 350*15.0)/1e6 = 0.0126 + 0.00525 = 0.01785 per req
    #   daily: 0.01785 * 10000 = 178.50, monthly: 178.50 * 30 = $5,355
    # doc-proc: (12400*3.0 + 340*15.0)/1e6 = 0.0372 + 0.0051 = 0.0423 per req
    #   daily: 0.0423 * 1000 = 42.30, monthly: 42.30 * 30 = $1,269
    # code-gen: (800*3.0 + 1200*15.0)/1e6 = 0.0024 + 0.018 = 0.0204 per req
    #   daily: 0.0204 * 5000 = 102.00, monthly: 102.00 * 30 = $3,060
