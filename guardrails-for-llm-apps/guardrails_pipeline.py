"""
Production guardrails pipeline — wraps an LLM call with configurable checks.

Blog post: https://dadops.dev/blog/guardrails-for-llm-apps/
Code Block 5.

No API key needed (uses mock LLM for testing). Runs standalone.

Blog claims:
  - Short-circuits on first failed check (no LLM tokens burned on blocked requests)
  - Logs every check's result for debugging false positives
  - Pipeline adds 50-200ms total overhead in production
"""
from dataclasses import dataclass, field
from typing import Callable, List
from time import perf_counter

from injection_detector import GuardrailResult


@dataclass
class GuardrailsPipeline:
    """Wraps an LLM call with configurable input/output guardrails.

    Usage:
        pipeline = GuardrailsPipeline(
            llm_fn=my_llm_call,
            input_checks=[check_injection, scan_pii],
            output_checks=[check_output_safety],
        )
        result = pipeline.run("Hello, can you help me?")
    """
    llm_fn: Callable[[str], str]
    input_checks: List[Callable] = field(default_factory=list)
    output_checks: List[Callable] = field(default_factory=list)
    fallback_response: str = "I'm sorry, I can't process that request. Please try again."

    def run(self, user_input: str) -> dict:
        log = {"input": user_input, "checks": [], "blocked": False}
        start = perf_counter()

        # ── Input guardrails ──────────────────────────────
        processed_input = user_input
        for check_fn in self.input_checks:
            result = check_fn(processed_input)

            # Handle PII scanner (returns tuple with cleaned text)
            if isinstance(result, tuple):
                result, processed_input, _ = result

            log["checks"].append({
                "stage": "input",
                "check": result.check_name,
                "passed": result.passed,
                "detail": result.detail,
            })

            if not result.passed:
                log["blocked"] = True
                log["response"] = self.fallback_response
                log["latency_ms"] = (perf_counter() - start) * 1000
                return log

        # ── LLM call ─────────────────────────────────────
        llm_response = self.llm_fn(processed_input)

        # ── Output guardrails ────────────────────────────
        for check_fn in self.output_checks:
            result = check_fn(llm_response)
            log["checks"].append({
                "stage": "output",
                "check": result.check_name,
                "passed": result.passed,
                "detail": result.detail,
            })

            if not result.passed:
                log["blocked"] = True
                log["response"] = self.fallback_response
                log["latency_ms"] = (perf_counter() - start) * 1000
                return log

        log["response"] = llm_response
        log["latency_ms"] = (perf_counter() - start) * 1000
        return log
