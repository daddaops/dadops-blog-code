"""
Evaluate base vs fine-tuned model on held-out test data.

Blog post: https://dadops.dev/blog/fine-tuning-llms/
Code Block 6.

REQUIRES: OpenAI API key (set OPENAI_API_KEY environment variable)

Blog claims:
  - Schema Match Rate: Base 82% -> Fine-tuned 99%
  - Field Accuracy: Base 74% -> Fine-tuned 91%
  - Prompt Tokens/Request: Base ~1,800 -> Fine-tuned ~120
  - Cost per Request: Base ~$0.001 -> Fine-tuned ~$0.00008
  - Latency: Base ~800ms -> Fine-tuned ~200ms
"""
import json
from openai import OpenAI

client = OpenAI()


def evaluate_model(model_name, test_data, use_few_shot=False):
    """Run a model on test data and measure accuracy."""
    correct_schema, correct_fields, total = 0, 0, 0
    required_fields = {"category", "priority", "platform", "feature", "sentiment"}

    few_shot_prefix = []
    if use_few_shot:
        # Base model needs examples in the prompt; fine-tuned model doesn't
        few_shot_prefix = [
            {"role": "user", "content": "Subject: Login broken\nBody: Can't login on Chrome."},
            {"role": "assistant", "content": '{"category":"bug","priority":"high","platform":"web","feature":"auth","sentiment":"frustrated"}'},
        ]

    for example in test_data:
        user_msg = [m for m in example["messages"] if m["role"] == "user"][0]
        expected = json.loads([m for m in example["messages"] if m["role"] == "assistant"][0]["content"])

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Extract structured data from support tickets. Return valid JSON with fields: category, priority, platform, feature, sentiment."},
                *few_shot_prefix,
                user_msg
            ]
        )

        try:
            output = json.loads(response.choices[0].message.content)
            if set(output.keys()) >= required_fields:
                correct_schema += 1
            matching = sum(1 for k in required_fields if output.get(k) == expected.get(k))
            correct_fields += matching / len(required_fields)
        except (json.JSONDecodeError, KeyError):
            pass  # Failed to produce valid JSON at all

        total += 1

    return {
        "schema_match": f"{correct_schema/total:.0%}",
        "field_accuracy": f"{correct_fields/total:.0%}",
        "total": total
    }


if __name__ == "__main__":
    # Load held-out test set
    with open("test_data.jsonl") as f:
        test_data = [json.loads(line) for line in f]

    base_results = evaluate_model("gpt-4o-mini", test_data, use_few_shot=True)
    ft_results = evaluate_model("ft:gpt-4o-mini:ticket-classifier:abc123", test_data)

    print("Base model (few-shot):", base_results)
    print("Fine-tuned model:    ", ft_results)
