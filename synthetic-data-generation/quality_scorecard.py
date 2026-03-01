"""
Quality scorecard for synthetic datasets.

Computes five metrics:
1. Label accuracy (via LLM judge)
2. Diversity (average pairwise embedding distance)
3. Difficulty distribution spread
4. Category coverage
5. Test set contamination risk

Requires: OPENAI_API_KEY environment variable.

From: https://dadops.dev/blog/synthetic-data-generation/
"""

import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

client = openai.OpenAI()


def get_embeddings(texts, model="text-embedding-3-small"):
    """Get embeddings for a list of texts."""
    resp = client.embeddings.create(input=texts, model=model)
    return np.array([e.embedding for e in resp.data])


def quality_scorecard(dataset, test_set=None, judge="gpt-4o"):
    """Compute comprehensive quality metrics for a synthetic dataset."""

    # 1. Label Accuracy — sample and verify with a judge model
    sample = random.sample(dataset, min(50, len(dataset)))
    correct = sum(
        1 for ex in sample
        if "correct" in client.chat.completions.create(
            model=judge,
            messages=[{"role": "user", "content":
                f'Is this label correct?\nText: "{ex["text"]}"\n'
                f'Label: {ex["label"]}\nReply: correct or incorrect'}]
        ).choices[0].message.content.lower()
    )
    label_accuracy = correct / len(sample)

    # 2. Diversity — average pairwise embedding distance
    embs = get_embeddings([ex["text"] for ex in dataset])
    pairwise = 1 - cosine_similarity(embs)
    idx = np.triu_indices(len(dataset), k=1)
    diversity = float(np.mean(pairwise[idx]))

    # 3. Difficulty Distribution — spread of coherence scores
    scores = [ex.get("coherence", 3) for ex in dataset]
    difficulty_spread = float(np.std(scores))

    # 4. Category Coverage
    categories = set(
        ex.get("label", ex.get("intent", "unknown")) for ex in dataset
    )

    # 5. Test Set Contamination Check
    contamination = 0.0
    if test_set:
        test_embs = get_embeddings([t["text"] for t in test_set])
        cross_sim = cosine_similarity(embs, test_embs)
        contamination = float(np.mean(cross_sim.max(axis=1) > 0.9))

    scorecard = {
        "Label Accuracy":     f"{label_accuracy:.0%}",
        "Diversity Score":    f"{diversity:.3f}",
        "Difficulty Spread":  f"{difficulty_spread:.2f}",
        "Category Coverage":  f"{len(categories)} categories",
        "Contamination Risk": f"{contamination:.1%} overlap",
    }

    print("\nData Quality Scorecard")
    print("=" * 44)
    for metric, value in scorecard.items():
        dots = "." * (32 - len(metric))
        print(f"  {metric} {dots} {value}")
    return scorecard
