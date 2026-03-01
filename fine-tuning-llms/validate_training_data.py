"""
Validate JSONL training data before uploading for fine-tuning.

Blog post: https://dadops.dev/blog/fine-tuning-llms/
Code Block 2.

Checks: valid JSON, correct role structure, valid assistant JSON output,
required fields present. No external dependencies needed.
"""
import json


def validate_training_data(filepath):
    """Validate JSONL training data before uploading."""
    errors, examples = [], []
    required_fields = {"category", "priority", "platform", "feature", "sentiment"}

    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {i}: Invalid JSON")
                continue

            msgs = obj.get("messages", [])
            roles = [m["role"] for m in msgs]
            if roles[0] != "system" or "user" not in roles or "assistant" not in roles:
                errors.append(f"Line {i}: Missing system/user/assistant role")
                continue

            # Check assistant output is valid JSON with required fields
            assistant_msg = [m for m in msgs if m["role"] == "assistant"][0]
            try:
                output = json.loads(assistant_msg["content"])
                missing = required_fields - set(output.keys())
                if missing:
                    errors.append(f"Line {i}: Missing fields: {missing}")
            except json.JSONDecodeError:
                errors.append(f"Line {i}: Assistant response is not valid JSON")

            examples.append(obj)

    print(f"Total examples: {len(examples)}")
    print(f"Errors found: {len(errors)}")
    for e in errors[:10]:
        print(f"  {e}")
    return len(errors) == 0


if __name__ == "__main__":
    print("=== Training Data Validation ===\n")
    valid = validate_training_data("training_data.jsonl")
    print(f"\nValidation: {'PASSED' if valid else 'FAILED'}")

    # Expected output:
    # Total examples: 3
    # Errors found: 0
    # Validation: PASSED
