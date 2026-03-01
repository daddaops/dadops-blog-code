"""
Full EvalHarness: combining all three evaluation levels.

Blog post: https://dadops.dev/blog/evaluating-llm-systems/
Code Blocks 3, 5, 6, and 7.

This is the reusable evaluation framework you can drop into any LLM project.
Demo uses mock components — replace with real LLM calls in production.
"""
import json
from dataclasses import dataclass, field
from typing import Optional


# ── Code Block 3: LLM-as-Judge ──

FAITHFULNESS_RUBRIC = """You are evaluating whether an AI response is faithful
to the provided context — meaning every claim in the response is supported by
the context.

PASS: Every factual claim in the response can be directly traced to the
provided context. The response does not add information beyond what the
context contains.

FAIL: The response contains at least one claim that is not supported by
the context, OR the response fabricates details not present in the context.

IMPORTANT: A short, accurate response is better than a long, embellished one.
Do not favor longer responses.

First explain your reasoning step by step, then give your verdict: PASS or FAIL.
"""


def llm_judge(input_text, response, context, rubric, call_llm):
    """Use a stronger model to grade a response against a rubric."""
    prompt = f"""{rubric}

INPUT: {input_text}
CONTEXT: {context}
RESPONSE: {response}

Reasoning and verdict:"""

    judgment = call_llm(prompt)  # your API call function
    passed = "PASS" in judgment.upper().split("FAIL")[0]  # check PASS before any FAIL
    return {"passed": passed, "reasoning": judgment}


# ── Code Block 5: Golden Dataset ──

@dataclass
class EvalCase:
    input: str
    expected: Optional[str] = None
    context: Optional[str] = None
    category: str = "general"
    metadata: dict = field(default_factory=dict)


def load_golden_dataset(path: str) -> list:
    """Load eval cases from a JSON file in your repo."""
    with open(path) as f:
        data = json.load(f)
    return [EvalCase(**case) for case in data]


# ── Code Block 6: EvalHarness ──

@dataclass
class EvalResult:
    case: EvalCase
    output: str
    scores: dict
    passed: bool


class EvalHarness:
    """Reusable evaluation harness for any LLM system."""

    def __init__(self, system_fn, metrics: list, judge_fn=None):
        """
        system_fn: callable(input, context=None) -> str
        metrics: list of dicts with 'name', 'fn', 'threshold'
        judge_fn: optional callable(prompt) -> str for LLM-as-judge
        """
        self.system_fn = system_fn
        self.metrics = metrics
        self.judge_fn = judge_fn

    def evaluate(self, cases: list) -> list:
        results = []
        for case in cases:
            output = self.system_fn(case.input, context=case.context)
            scores = {}

            for metric in self.metrics:
                scores[metric["name"]] = metric["fn"](
                    case=case, output=output, judge_fn=self.judge_fn
                )

            passed = all(
                scores[m["name"]] >= m["threshold"] for m in self.metrics
            )
            results.append(EvalResult(case=case, output=output,
                                      scores=scores, passed=passed))
        return results

    def summary(self, results: list) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        metric_avgs = {}
        for m in self.metrics:
            vals = [r.scores[m["name"]] for r in results]
            metric_avgs[m["name"]] = sum(vals) / len(vals)

        failures = [
            {"input": r.case.input[:80], "output": r.output[:200],
             "scores": r.scores}
            for r in results if not r.passed
        ]
        return {
            "total": total, "passed": passed,
            "pass_rate": f"{100 * passed / total:.1f}%",
            "metric_averages": metric_avgs,
            "failures": failures
        }


# ── Code Block 7: Metric Functions ──

def json_valid(case, output, **kw):
    """Level 1: does the output parse as JSON?"""
    try:
        json.loads(output)
        return 1.0
    except (json.JSONDecodeError, TypeError):
        return 0.0 if case.metadata.get("expects_json") else 1.0


def answer_contains_expected(case, output, **kw):
    """Level 1: does the answer contain the expected content?"""
    if case.expected is None:
        return 1.0  # no expected output to check
    return 1.0 if case.expected.lower() in output.lower() else 0.0


def faithfulness_judge(case, output, judge_fn=None, **kw):
    """Level 2: LLM-as-judge faithfulness check."""
    if judge_fn is None or not case.context:
        return 1.0  # skip if no judge or no context
    result = llm_judge(case.input, output, case.context,
                       FAITHFULNESS_RUBRIC, judge_fn)
    return 1.0 if result["passed"] else 0.0


# ── Demo with mock components ──

def mock_rag_system(input_text, context=None):
    """A mock RAG system that generates responses from context."""
    if context:
        # Simulate a system that mostly sticks to context
        if "refund" in input_text.lower():
            return "We offer a 30-day money-back guarantee on all plans."
        elif "cancel" in input_text.lower():
            return "You cannot cancel subscriptions."  # deliberate mismatch
        elif "discount" in input_text.lower():
            return "We offer a 50% discount for students."  # hallucination
        return f"Based on the context: {context[:100]}"
    return "I don't have enough context to answer that question."


def mock_judge(prompt):
    """A mock LLM judge that checks for obvious faithfulness issues."""
    prompt_lower = prompt.lower()
    # Simple heuristic: if the response adds claims not in context, FAIL
    if "cannot cancel" in prompt_lower and "cancel anytime" in prompt_lower:
        return "The response says 'cannot cancel' but context says 'cancel anytime'. FAIL"
    if "50% discount" in prompt_lower and "50%" not in prompt_lower.split("context:")[1].split("response:")[0]:
        return "The response mentions a 50% discount not found in context. FAIL"
    return "The response is consistent with the provided context. PASS"


if __name__ == "__main__":
    print("=== EvalHarness Demo ===\n")

    # Load golden dataset
    cases = load_golden_dataset("golden_dataset.json")
    print(f"Loaded {len(cases)} eval cases\n")

    # Build the harness
    harness = EvalHarness(
        system_fn=mock_rag_system,
        judge_fn=mock_judge,
        metrics=[
            {"name": "contains_expected", "fn": answer_contains_expected, "threshold": 0.5},
            {"name": "faithfulness", "fn": faithfulness_judge, "threshold": 0.5},
        ]
    )

    # Run evaluation
    results = harness.evaluate(cases)
    report = harness.summary(results)

    print(f"Pass rate: {report['pass_rate']}")
    print(f"Contains expected: {report['metric_averages']['contains_expected']:.0%}")
    print(f"Faithfulness: {report['metric_averages']['faithfulness']:.0%}")

    if report["failures"]:
        print(f"\nFailures ({len(report['failures'])}):")
        for fail in report["failures"]:
            print(f"  Input: {fail['input']}")
            print(f"  Scores: {fail['scores']}")
            print()
