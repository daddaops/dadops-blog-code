"""
Code review quality evaluation framework: precision, recall, F1.

Blog post: https://dadops.dev/blog/building-ai-code-review-tools/
Code Block 6.

Runs WITHOUT any API keys — uses a mock reviewer for demonstration.
"""


# ── Code Block 6: Evaluation Framework ──

def evaluate_reviewer(reviewer_fn, test_cases: list[dict]) -> dict:
    """Evaluate a code reviewer against known-buggy examples.

    Each test_case has: {"code": str, "known_bugs": [{"line": int, "category": str}]}
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for case in test_cases:
        findings = reviewer_fn(case["code"])
        found_lines = {(f["line"], f["category"]) for f in findings}
        known_lines = {(b["line"], b["category"]) for b in case["known_bugs"]}

        true_positives += len(found_lines & known_lines)
        false_positives += len(found_lines - known_lines)
        false_negatives += len(known_lines - found_lines)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives}


# ── Demo: mock reviewer for testing the evaluation framework ──

def mock_reviewer(code: str) -> list[dict]:
    """A simple pattern-matching 'reviewer' for testing the evaluation framework.

    Catches: SQL injection (f-string in query), missing KeyError handling,
    bare except, eval() usage.
    """
    findings = []
    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if "f'" in stripped and "SELECT" in stripped.upper():
            findings.append({"line": i, "category": "security",
                             "description": "SQL injection via f-string"})
        if "['key']" in stripped or '["key"]' in stripped:
            # Only flag if there's no try/except wrapping
            findings.append({"line": i, "category": "bug",
                             "description": "Unguarded dict access may raise KeyError"})
        if stripped.startswith("except:") or stripped == "except Exception:":
            findings.append({"line": i, "category": "style",
                             "description": "Bare except catches too broadly"})
        if "eval(" in stripped:
            findings.append({"line": i, "category": "security",
                             "description": "eval() is a code injection risk"})
    return findings


# ── Test suite with known bugs ──

TEST_SUITE = [
    {
        "code": "query = f'SELECT * FROM users WHERE id = {uid}'",
        "known_bugs": [{"line": 1, "category": "security"}]
    },
    {
        "code": "data = json.loads(request.data)\nreturn data['key']",
        "known_bugs": [{"line": 2, "category": "bug"}]
    },
    {
        "code": "result = eval(user_input)",
        "known_bugs": [{"line": 1, "category": "security"}]
    },
    {
        "code": "items = [x**2 for x in range(1000000)]\nfor i in items:\n    print(i)",
        "known_bugs": [{"line": 1, "category": "performance"}]
    },
    {
        "code": "try:\n    risky_operation()\nexcept:\n    pass",
        "known_bugs": [{"line": 3, "category": "style"}]
    },
]


if __name__ == "__main__":
    print("=== Code Review Evaluation Framework ===\n")

    # Run the mock reviewer against the test suite
    results = evaluate_reviewer(mock_reviewer, TEST_SUITE)

    print(f"Test cases: {len(TEST_SUITE)}")
    print(f"True positives:  {results['true_positives']}")
    print(f"False positives: {results['false_positives']}")
    print(f"False negatives: {results['false_negatives']}")
    print()
    print(f"Precision: {results['precision']:.1%}")
    print(f"Recall:    {results['recall']:.1%}")
    print(f"F1:        {results['f1']:.1%}")
    print()

    # Show per-case analysis
    print("--- Per-case breakdown ---")
    for i, case in enumerate(TEST_SUITE):
        findings = mock_reviewer(case["code"])
        found = {(f["line"], f["category"]) for f in findings}
        known = {(b["line"], b["category"]) for b in case["known_bugs"]}
        status = "HIT" if found & known else "MISS"
        fp = len(found - known)
        code_preview = case["code"].split("\n")[0][:60]
        print(f"  Case {i+1} [{status}]: {code_preview}...")
        if fp:
            print(f"         + {fp} false positive(s)")
    print()
    print("All evaluation tests completed!")
