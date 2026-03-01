"""
Code Block 1: Eval Harness — PromptEval framework.

From: https://dadops.dev/blog/prompt-engineering-systematic/

Defines TestCase, score_classification(), and PromptEval class.
The scoring logic can be tested standalone without an LLM API.
PromptEval.evaluate() requires a call_llm_fn to call an actual LLM.

No external dependencies required (stdlib only).
"""

from dataclasses import dataclass


@dataclass
class TestCase:
    """A single eval case: input text, expected category, expected issue."""
    input_text: str
    expected_category: str
    expected_issue_keywords: list[str]  # keywords that should appear in extraction


def score_classification(output: str, test_case: TestCase) -> dict:
    """Score an LLM output against expected results.

    Returns dict with:
      - category_correct: bool (exact match on category)
      - extraction_quality: float (0-1, based on keyword overlap)
      - parseable: bool (could we extract structured fields?)
    """
    lines = output.strip().split("\n")
    parsed_category = None
    parsed_issue = ""

    for line in lines:
        lower = line.lower().strip()
        if lower.startswith("category:"):
            parsed_category = lower.replace("category:", "").strip()
        elif lower.startswith("issue:"):
            parsed_issue = line.split(":", 1)[1].strip()

    category_correct = (
        parsed_category == test_case.expected_category.lower()
        if parsed_category else False
    )

    # keyword overlap score for extraction quality
    if parsed_issue and test_case.expected_issue_keywords:
        issue_lower = parsed_issue.lower()
        hits = sum(1 for kw in test_case.expected_issue_keywords if kw.lower() in issue_lower)
        extraction_quality = hits / len(test_case.expected_issue_keywords)
    else:
        extraction_quality = 0.0

    return {
        "category_correct": category_correct,
        "extraction_quality": extraction_quality,
        "parseable": parsed_category is not None,
    }


class PromptEval:
    """Run a prompt template against test cases and score the results."""

    def __init__(self, test_cases: list[TestCase], call_llm_fn):
        self.test_cases = test_cases
        self.call_llm = call_llm_fn  # function(system_prompt, user_msg) -> str

    def evaluate(self, system_prompt: str, runs_per_case: int = 3) -> dict:
        """Evaluate a prompt across all test cases, multiple runs each.

        Returns aggregate metrics: accuracy, parse_rate, avg_extraction,
        consistency (same category across runs), and total_tokens.
        """
        results = []
        total_tokens = 0

        for tc in self.test_cases:
            case_results = []
            for _ in range(runs_per_case):
                output = self.call_llm(system_prompt, tc.input_text)
                total_tokens += len(output.split()) * 1.3  # rough token estimate
                score = score_classification(output, tc)
                case_results.append(score)

            # Aggregate per-case: majority category, average extraction
            categories_correct = [r["category_correct"] for r in case_results]
            results.append({
                "category_correct": sum(categories_correct) / len(categories_correct) > 0.5,
                "extraction_quality": sum(r["extraction_quality"] for r in case_results) / len(case_results),
                "parseable": all(r["parseable"] for r in case_results),
                "consistent": len(set(str(r["category_correct"]) for r in case_results)) == 1,
            })

        n = len(results)
        return {
            "accuracy": sum(r["category_correct"] for r in results) / n,
            "parse_rate": sum(r["parseable"] for r in results) / n,
            "avg_extraction": sum(r["extraction_quality"] for r in results) / n,
            "consistency": sum(r["consistent"] for r in results) / n,
            "total_tokens": int(total_tokens),
        }


if __name__ == "__main__":
    print("=== Eval Harness — Scoring Logic Test ===\n")

    # Test the scoring function standalone (no LLM needed)
    tc = TestCase(
        input_text="I was charged twice for order #4821",
        expected_category="billing",
        expected_issue_keywords=["charged", "twice", "order"]
    )

    # Simulated correct LLM output
    good_output = "Category: billing\nIssue: Customer was charged twice for order #4821"
    score = score_classification(good_output, tc)
    print(f"Good output: {score}")

    # Simulated wrong category
    wrong_output = "Category: technical\nIssue: Charged twice for order"
    score = score_classification(wrong_output, tc)
    print(f"Wrong category: {score}")

    # Simulated unparseable output
    bad_output = "I think this is about billing. The customer was charged twice."
    score = score_classification(bad_output, tc)
    print(f"Unparseable: {score}")

    # Simulated partial extraction
    partial_output = "Category: billing\nIssue: Payment problem"
    score = score_classification(partial_output, tc)
    print(f"Partial extraction: {score}")
