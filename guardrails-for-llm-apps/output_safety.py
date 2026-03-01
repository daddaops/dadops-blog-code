"""
Content safety filter for LLM output.

Blog post: https://dadops.dev/blog/guardrails-for-llm-apps/
Code Block 4.

No API key needed. Runs standalone.

Blog claims:
  - Checks for refusal patterns (model declined when it shouldn't have)
  - Checks for off-topic content in context-specific mode
  - Length sanity check (responses < 10 chars are suspect)
"""
import re
from injection_detector import GuardrailResult

REFUSAL_PATTERNS = [
    r"i (?:can't|cannot|am unable to|won't|will not) (?:help|assist|provide|do that)",
    r"as an ai(?: language model)?",
    r"i (?:don't|do not) have (?:access|the ability)",
    r"it(?:'s| is) (?:not appropriate|inappropriate) for me",
]

OFF_TOPIC_MARKERS = [
    r"\b(?:political|religion|conspiracy|gambling|weapon)\b",
]

def check_output_safety(response: str, context: str = "customer_support") -> GuardrailResult:
    """Validate LLM output for refusals, off-topic content, and safety issues."""
    response_lower = response.lower()

    # Check for refusal patterns — the model declined when it shouldn't have
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, response_lower):
            return GuardrailResult(
                passed=False,
                check_name="output_safety",
                detail="Model refusal detected — may need prompt adjustment"
            )

    # Context-specific off-topic detection
    if context == "customer_support":
        for pattern in OFF_TOPIC_MARKERS:
            if re.search(pattern, response_lower):
                return GuardrailResult(
                    passed=False,
                    check_name="output_safety",
                    detail="Off-topic content detected in response"
                )

    # Length sanity check — absurdly short or long responses are suspect
    if len(response.strip()) < 10:
        return GuardrailResult(
            passed=False,
            check_name="output_safety",
            detail="Response suspiciously short"
        )

    return GuardrailResult(passed=True, check_name="output_safety")
