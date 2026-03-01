# Guardrails for LLM Apps — Verified Code

Code extracted from the DadOps blog post:
**[Guardrails for LLM Apps](https://daddaops.com/blog/guardrails-for-llm-apps/)**

## Scripts

| Script | Description | API Key? |
|--------|-------------|----------|
| `unprotected_llm.py` | The "before" picture — raw LLM call with no safety | Yes (OpenAI) |
| `injection_detector.py` | Regex-based prompt injection detection (8 patterns) | No |
| `pii_scanner.py` | PII detection and redaction (email, phone, SSN, CC, IP) | No |
| `output_safety.py` | Content safety filter for LLM output | No |
| `guardrails_pipeline.py` | Production middleware pipeline wrapping all checks | No |
| `verify_guardrails.py` | Integration test exercising all components | No |

## Quick Start

```bash
pip install -r requirements.txt
python verify_guardrails.py
```

All scripts except `unprotected_llm.py` run without any API keys.
