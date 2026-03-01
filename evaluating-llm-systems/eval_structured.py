"""
Level 1: Structured output evaluation — field-by-field comparison.

Blog post: https://dadops.dev/blog/evaluating-llm-systems/
Code Block 1.

Scores each field independently so you can see exactly which fields
the LLM gets right vs wrong.
"""
import json


def evaluate_structured_output(actual: dict, expected: dict) -> dict:
    """Score each field independently — a response can nail the name
    but hallucinate the email."""
    results = {}
    all_fields = set(expected.keys()) | set(actual.keys())

    for field in all_fields:
        if field not in actual:
            results[field] = {"status": "missing", "score": 0.0}
        elif field not in expected:
            results[field] = {"status": "extra", "score": 0.0}
        elif actual[field] == expected[field]:
            results[field] = {"status": "match", "score": 1.0}
        else:
            results[field] = {
                "status": "mismatch", "score": 0.0,
                "expected": expected[field], "actual": actual[field]
            }

    accuracy = sum(r["score"] for r in results.values()) / len(results)
    return {"field_results": results, "accuracy": accuracy}


if __name__ == "__main__":
    print("=== Structured Output Evaluation Demo ===\n")

    # Example 1: Receipt parser output (blog example)
    result = evaluate_structured_output(
        actual={"store": "Trader Joe's", "total": 47.82, "items": 12},
        expected={"store": "Trader Joe's", "total": 47.82, "items": 12, "date": "2026-02-20"}
    )
    print("Test 1: Receipt parser (missing date field)")
    print(f"  Accuracy: {result['accuracy']:.2f}")
    for field, info in result["field_results"].items():
        print(f"  {field}: {info['status']} (score={info['score']})")

    # Example 2: Perfect match
    result2 = evaluate_structured_output(
        actual={"name": "Alice", "age": 30},
        expected={"name": "Alice", "age": 30}
    )
    print(f"\nTest 2: Perfect match")
    print(f"  Accuracy: {result2['accuracy']:.2f}")

    # Example 3: Extra hallucinated field
    result3 = evaluate_structured_output(
        actual={"name": "Bob", "age": 25, "zodiac": "Leo"},
        expected={"name": "Bob", "age": 25}
    )
    print(f"\nTest 3: Extra hallucinated field")
    print(f"  Accuracy: {result3['accuracy']:.2f}")
    for field, info in result3["field_results"].items():
        print(f"  {field}: {info['status']} (score={info['score']})")

    # Example 4: Mismatch
    result4 = evaluate_structured_output(
        actual={"name": "Charlie", "total": 99.99},
        expected={"name": "Charlie", "total": 49.99}
    )
    print(f"\nTest 4: Value mismatch")
    print(f"  Accuracy: {result4['accuracy']:.2f}")
    for field, info in result4["field_results"].items():
        print(f"  {field}: {info['status']} (score={info['score']})")

    # Blog claims: accuracy 0.75 for 3/4 fields right
    # Expected output:
    # Test 1: Accuracy: 0.75
    # Test 2: Accuracy: 1.00
    # Test 3: Accuracy: 0.67
    # Test 4: Accuracy: 0.50
