import sys
import os
import numpy as np

# Import BiEncoder from script 02
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_mod = import_module("02_bi_encoder")
BiEncoder = _mod.BiEncoder

def mine_hard_negatives(retriever, queries, pos_docs, corpus, top_k=10):
    """
    Iterative hard negative mining: use the current model
    to find near-miss documents as hard negatives.
    """
    hard_negs = []
    for q, pos in zip(queries, pos_docs):
        results = retriever.search(q, corpus, top_k=top_k)
        # Filter out the actual positive document
        negs = [corpus[idx] for idx, _ in results if corpus[idx] != pos]
        hard_negs.append(negs[:3])  # keep top-3 hard negatives
    return hard_negs

# Training loop with iterative hard negative mining:
# Round 1: Train with random/in-batch negatives
# Round 2: Mine hard negatives with round-1 model, retrain
# Round 3: Mine harder negatives with round-2 model, retrain
# Each round, the negatives get harder and the model improves.
#
# ANCE repeats this loop until convergence, refreshing the
# negative index every few thousand training steps.

# Example mining iteration
queries = ["how to fix a leaking faucet",
           "best programming language for beginners",
           "symptoms of vitamin D deficiency"]
positives = ["plumbing repair: stopping drips from fixtures",
             "python is ideal for newcomers to coding",
             "fatigue and bone pain from low vitamin D levels"]
corpus = positives + [
    "kitchen renovation cost estimates 2024",
    "javascript framework comparison react vue angular",
    "common cold vs flu: how to tell the difference",
    "how to replace a bathroom faucet cartridge",
    "learning python: first steps in programming",
    "vitamin supplements: what you need to know",
]

# After a few rounds, the model learns that
# "replace a faucet cartridge" is NOT the same as "fix a leak"
# even though they share "faucet" -- that's the power of hard negatives

retriever = BiEncoder()
hard_negs = mine_hard_negatives(retriever, queries, positives, corpus)
for q, negs in zip(queries, hard_negs):
    print(f"\nQuery: {q}")
    print(f"  Hard negatives:")
    for neg in negs:
        print(f"    - {neg}")
