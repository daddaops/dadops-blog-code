"""Budget Tracker for cost control in multi-agent systems.

Tracks cumulative token spend across agent calls and halts
execution if the budget is exceeded.
"""
from dataclasses import dataclass, field

from agent_base import Agent, AgentResult

from llm_mock import call_llm  # noqa: F401

# Approximate costs per 1K tokens (input/output blended)
MODEL_COSTS = {
    "claude-sonnet-4-20250514": 0.009,
    "claude-haiku-4-5-20251001": 0.002,
    "gpt-4o": 0.008,
    "gpt-4o-mini": 0.001,
}

@dataclass
class BudgetTracker:
    """Halt execution if cumulative spend exceeds the budget."""
    max_budget_usd: float = 1.00
    spent_usd: float = 0.0
    calls: list = field(default_factory=list)

    def record(self, result: AgentResult, model: str) -> None:
        cost_per_k = MODEL_COSTS.get(model, 0.01)
        cost = (result.tokens_used / 1000) * cost_per_k
        self.spent_usd += cost
        self.calls.append({
            "agent": result.agent_name,
            "tokens": result.tokens_used,
            "cost_usd": round(cost, 4),
            "elapsed": round(result.elapsed_sec, 2),
        })

    def check(self) -> None:
        if self.spent_usd >= self.max_budget_usd:
            raise RuntimeError(
                f"Budget exceeded: ${self.spent_usd:.2f} "
                f"/ ${self.max_budget_usd:.2f}"
            )

    def summary(self) -> str:
        lines = [f"{'Agent':<20} {'Tokens':>8} {'Cost':>8} {'Time':>6}"]
        lines.append("-" * 46)
        for c in self.calls:
            lines.append(
                f"{c['agent']:<20} {c['tokens']:>8} "
                f"${c['cost_usd']:>6.4f} {c['elapsed']:>5.1f}s"
            )
        lines.append("-" * 46)
        lines.append(f"{'TOTAL':<20} "
            f"{sum(c['tokens'] for c in self.calls):>8} "
            f"${self.spent_usd:>6.4f}")
        return "\n".join(lines)


if __name__ == "__main__":
    tracker = BudgetTracker(max_budget_usd=0.50)

    # Simulate a few agent calls
    agent = Agent(
        name="Summarizer",
        system_prompt="You are a summarizer. Write concise summaries.",
    )

    for i in range(3):
        result = agent.run(f"Summarize document {i+1} about Monte Carlo methods.")
        tracker.record(result, model="claude-sonnet-4-20250514")
        tracker.check()
        print(f"Call {i+1}: ${tracker.spent_usd:.4f} spent")

    print(f"\n{tracker.summary()}")
