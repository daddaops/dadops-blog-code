"""
Code Block 6: Distributed trace builder for multi-step LLM pipelines.

From: https://dadops.dev/blog/llm-observability/

Lightweight distributed tracing with Span and Trace dataclasses.
Uses context managers to time spans and produces a human-readable
tree summary.

No external dependencies required.
"""

import time, uuid
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class Span:
    name: str
    span_id: str = field(
        default_factory=lambda: uuid.uuid4().hex[:8])
    start_ms: float = 0
    end_ms: float = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0
    attributes: dict = field(default_factory=dict)


@dataclass
class Trace:
    trace_id: str = field(
        default_factory=lambda: uuid.uuid4().hex[:16])
    spans: list = field(default_factory=list)

    @contextmanager
    def span(self, name):
        s = Span(name=name,
                 start_ms=time.perf_counter() * 1000)
        try:
            yield s
        finally:
            s.end_ms = time.perf_counter() * 1000
            self.spans.append(s)

    def summary(self):
        total_ms = sum(s.end_ms - s.start_ms
                       for s in self.spans)
        total_cost = sum(s.cost_usd for s in self.spans)
        total_tok  = sum(s.tokens_in + s.tokens_out
                         for s in self.spans)

        lines = [f"Trace {self.trace_id} \u2014 "
                 f"{total_ms:.0f}ms, ${total_cost:.4f}, "
                 f"{total_tok} tokens"]
        for i, s in enumerate(self.spans):
            dur = s.end_ms - s.start_ms
            pct = (dur / total_ms * 100) if total_ms else 0
            prefix = "\u2514\u2500" if i == len(self.spans) - 1 else "\u251c\u2500"
            lines.append(
                f"  {prefix} {s.name}: {dur:.0f}ms ({pct:.0f}%) "
                f"[{s.tokens_in}+{s.tokens_out} tok, "
                f"${s.cost_usd:.4f}]")
        return "\n".join(lines)


if __name__ == "__main__":
    print("=== Trace Builder ===\n")

    # Simulate a RAG pipeline with the exact numbers from the blog
    trace = Trace(trace_id="a3f2b1c9d4e5f678")

    with trace.span("embed query") as s:
        time.sleep(0.045)
        s.tokens_in = 12; s.tokens_out = 0; s.cost_usd = 0.0000

    with trace.span("vector search") as s:
        time.sleep(0.120)
        s.tokens_in = 0; s.tokens_out = 0; s.cost_usd = 0.0000

    with trace.span("rerank") as s:
        time.sleep(0.340)
        s.tokens_in = 2100; s.tokens_out = 50; s.cost_usd = 0.0003

    with trace.span("chat gpt-4o") as s:
        time.sleep(1.755)
        s.tokens_in = 3400; s.tokens_out = 350; s.cost_usd = 0.0042

    with trace.span("guardrail") as s:
        time.sleep(0.080)
        s.tokens_in = 160; s.tokens_out = 10; s.cost_usd = 0.0002

    print(trace.summary())
    print()

    # Verify blog claims:
    # Total tokens: 12+0 + 0+0 + 2100+50 + 3400+350 + 160+10 = 6082
    total_tok = (12+0) + (0+0) + (2100+50) + (3400+350) + (160+10)
    print(f"Expected total tokens: 6082, calculated: {total_tok}")

    # Total cost: 0+0+0.0003+0.0042+0.0002 = 0.0047
    total_cost = 0.0000 + 0.0000 + 0.0003 + 0.0042 + 0.0002
    print(f"Expected total cost: $0.0047, calculated: ${total_cost:.4f}")

    # Cost from generation: 0.0042/0.0047 = 89.4%
    gen_pct = 0.0042 / 0.0047 * 100
    print(f"Generation cost %: {gen_pct:.1f}% (blog claims 89%)")
