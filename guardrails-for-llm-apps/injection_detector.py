"""
Prompt injection detection using regex pattern matching.

Blog post: https://dadops.dev/blog/guardrails-for-llm-apps/
Code Block 2.

No API key needed. Runs standalone.

Blog claims:
  - Catches roughly 60-70% of naive injection attacks
  - Runs in <1ms (O(n) regex scan)
  - 8 injection patterns ordered by specificity
"""
import re
from dataclasses import dataclass

@dataclass
class GuardrailResult:
    passed: bool
    check_name: str
    detail: str = ""

# Patterns that signal injection attempts — ordered by specificity
INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
     "instruction override"),
    (r"(reveal|show|display|output|print)\s+(the\s+)?(system\s+prompt|instructions|rules)",
     "system prompt extraction"),
    (r"you\s+are\s+now\s+(?!going)",
     "role reassignment"),
    (r"pretend\s+(you\s+are|to\s+be|you're)",
     "role-play attack"),
    (r"do\s+not\s+follow\s+(your|the|any)\s+(rules|instructions|guidelines)",
     "rule bypass"),
    (r"\bDAN\b.*\bdo\s+anything\b|\bdo\s+anything\b.*\bDAN\b",
     "DAN jailbreak"),
    (r"(system|admin|developer)\s*:\s*",
     "fake role prefix"),
    (r"<\s*/?\s*system\s*>",
     "XML tag injection"),
]

def check_injection(text: str) -> GuardrailResult:
    """Scan for common prompt injection patterns. O(n) regex scan, <1ms typical."""
    text_lower = text.lower()
    for pattern, label in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return GuardrailResult(
                passed=False,
                check_name="injection_detector",
                detail=f"Matched pattern: {label}"
            )
    return GuardrailResult(passed=True, check_name="injection_detector")
