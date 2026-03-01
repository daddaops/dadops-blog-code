"""
Prompt Compressor — applies three compression techniques to reduce token costs.

From: https://dadops.dev/blog/llm-cost-optimization/
Code Block 2: "Prompt Optimization — The Highest-ROI Change"

Techniques: instruction deduplication, example pruning, format compression.
Blog claims 30-50% token reduction is typical.
"""

import re
from dataclasses import dataclass


@dataclass
class CompressionResult:
    original_tokens: int
    compressed_tokens: int
    savings_pct: float
    compressed_text: str


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def compress_prompt(system_prompt: str,
                    examples: list[str],
                    max_examples: int = 3) -> CompressionResult:
    """Apply three compression techniques to a prompt."""
    original = system_prompt + "\n".join(examples)
    orig_tokens = estimate_tokens(original)

    # 1. Instruction deduplication — strip repeated phrases
    lines = system_prompt.split("\n")
    seen = set()
    deduped = []
    for line in lines:
        normalized = line.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(line)
    compressed_sys = "\n".join(deduped)

    # 2. Example pruning — keep only max_examples
    pruned_examples = examples[:max_examples]

    # 3. Format compression — shorten verbose JSON specs
    compressed_sys = re.sub(
        r"[Pp]lease respond with a JSON object containing[^.]+\.",
        lambda m: _compress_json_spec(m.group(0)),
        compressed_sys,
    )

    result = compressed_sys + "\n".join(pruned_examples)
    comp_tokens = estimate_tokens(result)
    savings = (1 - comp_tokens / orig_tokens) * 100 if orig_tokens else 0

    return CompressionResult(orig_tokens, comp_tokens,
                             savings, result)


def _compress_json_spec(verbose: str) -> str:
    """Turn verbose JSON field descriptions into terse specs."""
    fields = re.findall(r"(\w+)\s*\((?:a\s+)?(\w+)\)", verbose)
    if fields:
        spec = ", ".join(f"{n}: {t}" for n, t in fields)
        return "Return JSON: {" + spec + "}."
    return verbose


if __name__ == "__main__":
    # Demo with a realistic system prompt that has redundancy
    system_prompt = """You are a helpful customer support assistant.
You must always be polite and professional.
Always respond in a helpful manner.
You must always be polite and professional.
When the customer asks a question, provide a clear answer.
Always respond in a helpful manner.
Please respond with a JSON object containing summary (a string), sentiment (a string), and priority (an integer).
Make sure to include all relevant details in your response.
You are a helpful customer support assistant."""

    examples = [
        "Example 1: Customer asks about return policy. Response: Our return policy allows returns within 30 days of purchase with original receipt.",
        "Example 2: Customer complains about shipping delay. Response: I apologize for the delay. Let me check the status of your order.",
        "Example 3: Customer wants to upgrade subscription. Response: I'd be happy to help you upgrade. Here are the available plans.",
        "Example 4: Customer reports a bug. Response: Thank you for reporting this. I'll escalate to our engineering team.",
        "Example 5: Customer asks about pricing. Response: Here are our current pricing tiers for the service.",
        "Example 6: Customer wants a refund. Response: I understand your frustration. Let me process that refund for you.",
    ]

    result = compress_prompt(system_prompt, examples, max_examples=3)

    print("=== Prompt Compression Results ===")
    print(f"  Original tokens:   {result.original_tokens}")
    print(f"  Compressed tokens: {result.compressed_tokens}")
    print(f"  Savings:           {result.savings_pct:.1f}%")
    print(f"\n  Blog claims: 30-50% compression typical")
    print(f"  Actual:      {result.savings_pct:.1f}% compression")

    # Verify blog savings math:
    # 1,200 tokens saved × 10,000 requests × $3/M = $36/day = $1,080/month
    print(f"\n=== Blog Savings Math ===")
    tokens_saved = 1200
    daily_requests = 10_000
    daily_savings = tokens_saved * daily_requests * 3.0 / 1e6
    monthly_savings = daily_savings * 30
    print(f"  1,200 tokens × 10K requests × $3/M = ${daily_savings:.0f}/day")
    print(f"  Monthly: ${monthly_savings:.0f} (blog claims: $1,080/month)")
