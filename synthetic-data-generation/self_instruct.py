"""
Self-Instruct: generate labeled training data from a task description alone.

Uses high temperature + negative example feedback loop for diversity.
No seed data needed — just a well-written task description.

Requires: OPENAI_API_KEY environment variable.

From: https://dadops.dev/blog/synthetic-data-generation/
"""

import openai, json, random
from collections import Counter

client = openai.OpenAI()

TASK = """Generate diverse sentiment analysis training examples.
Each example: a JSON object with "text" and "label" fields.
Labels: positive, negative, neutral.
Vary length (5-50 words), domain (products, food, travel,
services, entertainment), and difficulty (obvious to subtle)."""


def self_instruct(task_desc, target=100, batch_size=5):
    """Generate labeled examples from a task description alone."""
    pool, seen = [], set()

    for _ in range(target // batch_size):
        # Show recent examples as negative examples for diversity
        recent = json.dumps(pool[-6:], indent=2) if pool else "[]"

        resp = client.chat.completions.create(
            model="gpt-4o-mini",   # Cheap model for bulk generation
            temperature=1.0,       # High temp = more diversity
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": task_desc},
                {"role": "user", "content": f"""Generate {batch_size} NEW examples.
They must be DIFFERENT from these existing ones:
{recent}

Mix: sarcasm, subtle opinions, mixed feelings, obvious cases.
Return JSON with an "examples" array."""}
            ]
        )

        batch = json.loads(resp.choices[0].message.content)
        for ex in batch.get("examples", []):
            key = ex["text"].strip().lower()
            if key not in seen:
                seen.add(key)
                pool.append(ex)

    labels = Counter(ex["label"] for ex in pool)
    print(f"Generated {len(pool)} examples: {dict(labels)}")
    return pool


if __name__ == "__main__":
    # Generate 100 sentiment examples from scratch — no labeled data needed
    examples = self_instruct(TASK, target=100)
    # Cost: ~$0.03 with gpt-4o-mini (Feb 2026 pricing)

    # Save to file
    with open("output/self_instruct_examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} examples to output/self_instruct_examples.json")
