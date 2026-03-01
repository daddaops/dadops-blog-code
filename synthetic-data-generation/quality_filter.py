"""
Quality filtering pipeline for synthetic data.

Four-stage filter:
1. Coherence — is the example well-formed? (LLM judge)
2. Label verification — does the label match the text? (LLM judge)
3. Embedding-based deduplication — remove near-copies
4. Difficulty calibration — remove what a baseline model already handles

Requires: OPENAI_API_KEY environment variable.

From: https://dadops.dev/blog/synthetic-data-generation/
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

client = openai.OpenAI()


def get_embeddings(texts, model="text-embedding-3-small"):
    """Get embeddings for a list of texts."""
    resp = client.embeddings.create(input=texts, model=model)
    return np.array([e.embedding for e in resp.data])


def quality_filter(examples, model="gpt-4o-mini"):
    """Four-stage quality filter for synthetic data."""
    print(f"Starting with {len(examples)} examples")

    # Stage 1: Coherence — is the example well-formed?
    coherent = []
    for ex in examples:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content":
                f'Rate this example\'s coherence 1-5.\n'
                f'Text: "{ex["text"]}"\nReply with just the number.'}]
        )
        score = int(resp.choices[0].message.content.strip())
        if score >= 3:
            ex["coherence"] = score
            coherent.append(ex)
    print(f"  After coherence filter: {len(coherent)}")

    # Stage 2: Label verification — does output match input?
    verified = []
    for ex in coherent:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content":
                f'Is this label correct?\nText: "{ex["text"]}"\n'
                f'Label: "{ex["label"]}"\nReply: correct or incorrect'}]
        )
        if "correct" in resp.choices[0].message.content.lower().split():
            verified.append(ex)
    print(f"  After label verification: {len(verified)}")

    # Stage 3: Embedding-based deduplication
    embs = get_embeddings([ex["text"] for ex in verified])
    sim_matrix = cosine_similarity(embs)
    np.fill_diagonal(sim_matrix, 0)
    keep = [True] * len(verified)
    for i in range(len(verified)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(verified)):
            if sim_matrix[i][j] > 0.95:
                keep[j] = False  # Mark duplicate for removal
    deduped = [ex for ex, k in zip(verified, keep) if k]
    print(f"  After deduplication: {len(deduped)}")

    # Stage 4: Difficulty calibration — remove too-easy examples
    final = []
    for ex in deduped:
        baseline = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": ex["text"]}]
        )
        answer = baseline.choices[0].message.content
        if not answer.strip().lower().startswith(ex["label"].lower()):
            final.append(ex)  # Keep what the baseline gets wrong
    print(f"  After difficulty filter: {len(final)}")
    return final


# Typical result: 1000 raw -> 850 coherent -> 780 verified
#   -> 650 deduped -> 550 difficulty-filtered
