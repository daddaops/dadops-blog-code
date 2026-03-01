"""
Code Block 3: Full V5 Prompt + Prompt Version Comparison Runner.

From: https://dadops.dev/blog/prompt-engineering-systematic/

Contains the complete V5 system prompt and run_prompt_versions() function.
Requires an LLM API to reproduce the benchmark results — SKIP for execution.
The prompt string itself is just a constant and can be inspected directly.

No external dependencies required (stdlib only + eval_harness from Block 1).
"""

from eval_harness import PromptEval

SYSTEM_PROMPT_V5 = """You are a senior customer support analyst at an e-commerce company.
You classify tickets for an automated routing system. Your output is parsed programmatically.

CLASSIFICATION CATEGORIES: billing, technical, account, shipping, returns, other

RULES:
- If a ticket spans multiple categories, choose the primary one based on
  what the customer needs resolved first.
- Do not classify as "other" unless none of the five specific categories apply.
- If genuinely uncertain, respond with category "uncertain".

CONSTRAINTS:
- Do not include explanations or reasoning.
- Do not apologize or hedge ("I think", "It seems").
- Do not add any text outside the specified format.
- Do not ask clarifying questions — use your best judgment.

EXAMPLES:
---
Input: "I was charged twice for order #4821 and the second charge is still pending"
<category>billing</category>
<issue>Double charge on order #4821 with pending duplicate</issue>
---
Input: "The app crashes every time I try to add items to my cart on iOS 17"
<category>technical</category>
<issue>App crash on iOS 17 when adding items to cart</issue>
---
Input: "I returned the shoes two weeks ago and still haven't gotten my money back, also my account shows the wrong email"
<category>returns</category>
<issue>Refund not received two weeks after shoe return</issue>

RESPONSE FORMAT (respond with ONLY these two XML lines):
<category>[one of: billing, technical, account, shipping, returns, other, uncertain]</category>
<issue>[one sentence describing the key customer issue]</issue>"""


def run_prompt_versions(eval_harness: PromptEval) -> None:
    """Run all 5 prompt versions and print a comparison table.

    NOTE: Requires a working LLM API via eval_harness.call_llm.
    """
    prompts = {
        "V1 Baseline": "You are a helpful assistant. Classify this support ticket.",
        "V2 + Role": """You are a senior customer support analyst. Classify tickets into:
billing, technical, account, shipping, returns, other.
Respond with Category: [cat] and Issue: [one sentence].""",
        "V3 + FewShot": """...""",  # V2 + 3 diverse examples (truncated for display)
        "V4 + Structure": """...""",  # V3 + XML tag format
        "V5 + Constraints": SYSTEM_PROMPT_V5,
    }

    print(f"{'Version':<20} {'Accuracy':>8} {'Parse':>8} {'Tokens':>8} {'Cost/1K':>10}")
    print("-" * 58)

    for name, prompt in prompts.items():
        results = eval_harness.evaluate(prompt, runs_per_case=3)
        cost = results["total_tokens"] * 0.30 / 1_000_000  # rough output cost
        print(f"{name:<20} {results['accuracy']:>7.0%} {results['parse_rate']:>7.0%} "
              f"{results['total_tokens']:>7d} ${cost:>8.4f}")


if __name__ == "__main__":
    print("=== Prompt Version Comparison ===\n")
    print("This script requires an LLM API key to run the full benchmark.")
    print("The V5 system prompt is defined as SYSTEM_PROMPT_V5 constant.")
    print(f"\nV5 prompt length: {len(SYSTEM_PROMPT_V5)} characters")
    print(f"V5 prompt lines: {len(SYSTEM_PROMPT_V5.strip().splitlines())}")
    print("\nClaimed results (from blog, require LLM API to verify):")
    print(f"{'Version':<20} {'Accuracy':>8} {'Parse':>8} {'Tokens':>8}")
    print("-" * 48)
    print(f"{'V1 Baseline':<20} {'62%':>8} {'70%':>8} {'5100':>8}")
    print(f"{'V2 + Role':<20} {'91%':>8} {'97%':>8} {'2100':>8}")
    print(f"{'V3 + FewShot':<20} {'94%':>8} {'97%':>8} {'2400':>8}")
    print(f"{'V4 + Structure':<20} {'94%':>8} {'99%':>8} {'2200':>8}")
    print(f"{'V5 + Constraints':<20} {'96%':>8} {'99%':>8} {'1680':>8}")
