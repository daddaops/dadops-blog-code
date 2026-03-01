"""
Code Block 4: LLM judge-based router using GPT-4o-mini.

From: https://dadops.dev/blog/llm-model-routing/

Uses a cheap LLM (GPT-4o-mini at $0.15/M input tokens) to classify
query complexity before routing to the appropriate tier.

Requires: openai
API key required: OPENAI_API_KEY
"""

import json
from openai import OpenAI

client = OpenAI()

JUDGE_PROMPT = """You are a query complexity classifier. Given a user query,
rate its complexity and pick the minimum model tier needed to answer it well.

Tiers:
- Tier 1: Simple lookups, factual questions, extraction, classification
- Tier 2: Moderate reasoning, summarization, content generation, code writing
- Tier 3: Complex multi-step reasoning, ambiguous problems, deep analysis

Return JSON only: {"tier": 1|2|3, "reason": "one sentence explanation"}"""


def judge_route(query: str) -> dict:
    """Use a cheap LLM to classify query complexity before routing."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # cheap judge — $0.15/M input tokens
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"},
        max_tokens=80,
        temperature=0
    )
    result = json.loads(response.choices[0].message.content)
    return result   # {"tier": 2, "reason": "Requires summarization..."}


if __name__ == "__main__":
    print("=== LLM Judge Router ===")
    print("Requires OPENAI_API_KEY environment variable.\n")

    try:
        decision = judge_route("Compare three approaches to database sharding "
                               "and recommend one for our 50TB analytics workload")
        print(decision)
        # Expected: {"tier": 3, "reason": "Multi-approach comparison with context-specific recommendation"}
    except Exception as e:
        print(f"SKIP: {e}")
