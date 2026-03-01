"""
PII detection and redaction using regex patterns.

Blog post: https://dadops.dev/blog/guardrails-for-llm-apps/
Code Block 3.

No API key needed. Runs standalone.

Blog claims:
  - Catches roughly 85% of common PII patterns
  - 5 PII types: email, phone_us, ssn, credit_card, ip_address
  - Redacts and continues (passed=True even when PII found)
"""
import re
from typing import Tuple
from injection_detector import GuardrailResult

PII_PATTERNS = {
    "email":       (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
                    "[EMAIL_REDACTED]"),
    "phone_us":    (r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
                    "[PHONE_REDACTED]"),
    "ssn":         (r"\b\d{3}-\d{2}-\d{4}\b",
                    "[SSN_REDACTED]"),
    "credit_card": (r"\b(?:\d[ -]*?){13,19}\b",
                    "[CC_REDACTED]"),
    "ip_address":  (r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                    "[IP_REDACTED]"),
}

def scan_pii(text: str) -> Tuple[GuardrailResult, str, dict]:
    """Detect and redact PII. Returns (result, cleaned_text, found_pii_types)."""
    cleaned = text
    found = {}

    for pii_type, (pattern, placeholder) in PII_PATTERNS.items():
        matches = re.findall(pattern, cleaned)
        if matches:
            found[pii_type] = len(matches)
            cleaned = re.sub(pattern, placeholder, cleaned)

    if found:
        detail = ", ".join(f"{k}: {v} found" for k, v in found.items())
        return (
            GuardrailResult(passed=True, check_name="pii_scanner",
                            detail=f"Redacted: {detail}"),
            cleaned,
            found
        )

    return (
        GuardrailResult(passed=True, check_name="pii_scanner"),
        text,
        {}
    )
