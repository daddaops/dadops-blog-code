"""
Token Budget Controller — per-feature token budgets with graceful degradation.

From: https://dadops.dev/blog/llm-cost-optimization/
Code Block 5: "Token Budget Management"

Enforces soft limits (model downgrade at 80%), reduced output (90%),
and rejection (95%) with critical-request bypass.
"""

import time
from dataclasses import dataclass
from enum import Enum


class BudgetAction(Enum):
    ALLOW = "allow"
    DOWNGRADE = "downgrade_model"  # switch to cheaper model
    REDUCE = "reduce_max_tokens"   # cap output length
    REJECT = "reject"              # drop non-critical requests


@dataclass
class FeatureBudget:
    name: str
    monthly_limit_tokens: int
    used_tokens: int = 0
    reset_at: float = 0.0

    @property
    def usage_pct(self) -> float:
        return (self.used_tokens / self.monthly_limit_tokens * 100
                if self.monthly_limit_tokens else 0)


class TokenBudgetController:
    """Per-feature token budgets with graceful degradation."""

    def __init__(self):
        self.budgets: dict[str, FeatureBudget] = {}

    def register(self, feature: str, monthly_tokens: int):
        self.budgets[feature] = FeatureBudget(
            name=feature,
            monthly_limit_tokens=monthly_tokens,
            reset_at=time.time() + 30 * 86400,
        )

    def check(self, feature: str, tokens: int,
              critical: bool = False) -> BudgetAction:
        budget = self.budgets.get(feature)
        if not budget:
            return BudgetAction.ALLOW

        # Auto-reset on new billing period
        if time.time() > budget.reset_at:
            budget.used_tokens = 0
            budget.reset_at = time.time() + 30 * 86400

        projected = ((budget.used_tokens + tokens)
                     / budget.monthly_limit_tokens * 100
                     if budget.monthly_limit_tokens else 0)
        if projected < 80:
            return BudgetAction.ALLOW
        elif projected < 90:
            return BudgetAction.DOWNGRADE
        elif projected < 95:
            return BudgetAction.REDUCE
        elif critical:
            return BudgetAction.ALLOW  # critical always passes
        else:
            return BudgetAction.REJECT

    def record(self, feature: str, tokens_used: int):
        if feature in self.budgets:
            self.budgets[feature].used_tokens += tokens_used

    def dashboard(self) -> str:
        lines = ["Feature Budget Dashboard", "=" * 50]
        for b in sorted(self.budgets.values(),
                        key=lambda x: -x.usage_pct):
            bar_len = int(b.usage_pct / 5)
            bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
            status = ("\U0001f534" if b.usage_pct >= 95
                      else "\U0001f7e1" if b.usage_pct >= 80
                      else "\U0001f7e2")
            lines.append(
                f"  {status} {b.name:<20} [{bar}] "
                f"{b.usage_pct:.1f}%"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    controller = TokenBudgetController()

    # Register features with monthly token budgets
    controller.register("chatbot", 100_000_000)       # 100M tokens
    controller.register("document-processor", 50_000_000)  # 50M tokens
    controller.register("code-generator", 30_000_000)  # 30M tokens

    # Simulate usage — chatbot at 85%, doc-proc at 40%, code-gen at 96%
    controller.budgets["chatbot"].used_tokens = 85_000_000
    controller.budgets["document-processor"].used_tokens = 20_000_000
    controller.budgets["code-generator"].used_tokens = 28_800_000

    print("=== Token Budget Controller Demo ===\n")
    print(controller.dashboard())

    # Test threshold actions
    print("\n=== Action Tests ===")
    test_cases = [
        ("chatbot", 1_000_000, False, "85% + 1M → should DOWNGRADE"),
        ("document-processor", 500_000, False, "40% + 500K → should ALLOW"),
        ("code-generator", 100_000, False, "96% + 100K → should REJECT"),
        ("code-generator", 100_000, True, "96% + 100K critical → should ALLOW"),
    ]

    for feature, tokens, critical, description in test_cases:
        action = controller.check(feature, tokens, critical)
        print(f"  {description}: {action.value}")
