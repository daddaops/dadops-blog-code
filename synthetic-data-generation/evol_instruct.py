"""
Evol-Instruct: evolve simple examples into progressively harder variants.

Applies five evolution operators (add constraints, deepen reasoning,
increase specificity, inject ambiguity, require tradeoffs) across
multiple rounds to create a difficulty spectrum.

Requires: OPENAI_API_KEY environment variable.

From: https://dadops.dev/blog/synthetic-data-generation/
"""

import json, random
import openai

client = openai.OpenAI()

EVOLUTION_OPS = {
    "add_constraints": "Add 2-3 specific constraints or conditions.",
    "deepen_reasoning": "Require multi-step reasoning or comparison.",
    "increase_specificity": "Add domain-specific context and details.",
    "inject_ambiguity": "Make it require clarification or interpretation.",
    "require_tradeoffs": "Reframe so the answer weighs pros and cons.",
}


def evolve_example(example, op_name, op_prompt):
    """Apply one evolution operator to increase difficulty."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": f"""Evolve this Q&A to be harder.

Original Question: {example["question"]}
Original Answer: {example["answer"]}

Evolution strategy: {op_prompt}
Return JSON with "question" and "answer" fields.
The evolved question must be harder but still answerable."""}]
    )
    return json.loads(resp.choices[0].message.content)


def evol_instruct(simple_examples, rounds=2):
    """Evolve simple examples into progressively harder variants."""
    all_examples = list(simple_examples)  # Keep originals
    current = simple_examples

    for round_num in range(rounds):
        evolved_batch = []
        operators = list(EVOLUTION_OPS.items())

        for ex in current:
            op_name, op_prompt = random.choice(operators)
            evolved = evolve_example(ex, op_name, op_prompt)
            evolved["round"] = round_num + 1
            evolved["operator"] = op_name
            evolved_batch.append(evolved)

        # Quality gate: reject if too similar to original
        for orig, evol in zip(current, evolved_batch):
            orig_words = set(orig["question"].lower().split())
            evol_words = set(evol["question"].lower().split())
            overlap = len(orig_words & evol_words) / len(orig_words | evol_words)
            if overlap < 0.7:  # Sufficiently different
                all_examples.append(evol)

        current = evolved_batch  # Feed evolved into next round

    print(f"Evolved {len(simple_examples)} -> {len(all_examples)} examples")
    return all_examples


if __name__ == "__main__":
    # Small demo with 3 simple Q&A pairs
    simple = [
        {"question": "What is machine learning?", "answer": "Machine learning is a subset of AI where models learn patterns from data."},
        {"question": "What is overfitting?", "answer": "Overfitting is when a model learns noise in training data and performs poorly on new data."},
        {"question": "What is a neural network?", "answer": "A neural network is layers of interconnected nodes that transform inputs into outputs."},
    ]

    evolved = evol_instruct(simple, rounds=2)

    with open("output/evol_instruct_examples.json", "w") as f:
        json.dump(evolved, f, indent=2)
    print(f"Saved {len(evolved)} examples to output/evol_instruct_examples.json")
