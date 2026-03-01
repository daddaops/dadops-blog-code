"""
Few-Shot Amplification: scale seed examples into a larger dataset.

Uses stratified random sampling from seed categories to maintain
diversity while grounding generation in real examples.

Requires: OPENAI_API_KEY environment variable.

From: https://dadops.dev/blog/synthetic-data-generation/
"""

import random, json
import openai

client = openai.OpenAI()

SEEDS = [
    {"text": "I need to change my delivery address", "intent": "address_change"},
    {"text": "Where's my package? It's been two weeks", "intent": "order_status"},
    {"text": "Can I get a refund for the broken item?", "intent": "refund_request"},
    {"text": "How do I reset my password?", "intent": "account_help"},
    {"text": "Do you ship internationally?", "intent": "shipping_info"},
    {"text": "I want to cancel my subscription", "intent": "cancellation"},
    # ... imagine 14 more spanning all intent categories
]
INTENTS = list(set(s["intent"] for s in SEEDS))


def few_shot_amplify(seeds, target=500, shots=3):
    """Scale seed examples into a larger dataset via few-shot prompting."""
    synthetic = []

    for i in range(target // 5):
        # Stratified sampling: pick seeds from different intents
        sampled = []
        for _ in range(shots):
            intent = random.choice(INTENTS)
            candidates = [s for s in seeds if s["intent"] == intent]
            sampled.append(random.choice(candidates))

        examples_block = "\n".join(
            f'  "{s["text"]}" -> {s["intent"]}' for s in sampled
        )

        resp = client.chat.completions.create(
            model="gpt-4o",       # Stronger model for quality
            temperature=0.9,
            messages=[{"role": "user", "content": f"""Real customer messages:
{examples_block}

Generate 5 NEW messages with correct intents.
Same style and difficulty. Valid intents: {INTENTS}
Return a JSON array of {{"text": "...", "intent": "..."}} objects."""}]
        )

        batch = json.loads(resp.choices[0].message.content)
        items = batch if isinstance(batch, list) else batch.get("examples", [])
        for ex in items:
            if ex.get("intent") in INTENTS:
                # Novelty check: how different from nearest seed?
                seed_words = [set(s["text"].lower().split()) for s in seeds]
                ex_words = set(ex["text"].lower().split())
                max_overlap = max(
                    len(ex_words & sw) / max(len(ex_words), 1)
                    for sw in seed_words
                )
                ex["novelty"] = round(1.0 - max_overlap, 2)
                synthetic.append(ex)

    avg_novelty = sum(e["novelty"] for e in synthetic) / len(synthetic)
    print(f"Generated {len(synthetic)} examples (avg novelty: {avg_novelty:.2f})")
    return synthetic


if __name__ == "__main__":
    synthetic = few_shot_amplify(SEEDS, target=50)  # Small demo run

    with open("output/few_shot_examples.json", "w") as f:
        json.dump(synthetic, f, indent=2)
    print(f"Saved {len(synthetic)} examples to output/few_shot_examples.json")
