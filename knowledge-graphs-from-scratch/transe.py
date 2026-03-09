import numpy as np
from graph_basics import triples

def train_transe(triples, entities, relations, dim=20, lr=0.01,
                 margin=1.0, epochs=500):
    """Train TransE: learn embeddings where h + r ≈ t."""
    rng = np.random.RandomState(42)
    ent_emb = rng.randn(len(entities), dim)
    ent_emb /= np.linalg.norm(ent_emb, axis=1, keepdims=True)
    rel_emb = rng.randn(len(relations), dim)
    rel_emb /= np.linalg.norm(rel_emb, axis=1, keepdims=True)

    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}
    ent_list = list(entities)

    for epoch in range(epochs):
        for h, r, t in triples:
            hi, ri, ti = ent2id[h], rel2id[r], ent2id[t]
            # Corrupt: replace head or tail randomly
            ci = ent2id[rng.choice(ent_list)]
            d_pos = np.linalg.norm(ent_emb[hi] + rel_emb[ri] - ent_emb[ti])
            if rng.rand() < 0.5:
                d_neg = np.linalg.norm(ent_emb[ci] + rel_emb[ri] - ent_emb[ti])
            else:
                d_neg = np.linalg.norm(ent_emb[hi] + rel_emb[ri] - ent_emb[ci])

            loss = max(0, margin + d_pos - d_neg)
            if loss > 0:
                grad = (ent_emb[hi] + rel_emb[ri] - ent_emb[ti])
                grad /= (np.linalg.norm(grad) + 1e-8)
                ent_emb[hi] -= lr * grad
                rel_emb[ri] -= lr * grad
                ent_emb[ti] += lr * grad

        ent_emb /= np.linalg.norm(ent_emb, axis=1, keepdims=True)

    return ent_emb, rel_emb, ent2id, rel2id

entities = {"Einstein", "Curie", "Bohr", "Physics", "Chemistry",
            "Nobel_Physics", "Nobel_Chemistry", "Ulm", "Warsaw", "Copenhagen"}
rels = {"field", "won", "born_in"}

all_entities = entities | {"Germany", "Poland", "Denmark"}
all_rels = rels | {"located_in"}

def get_embeddings():
    return train_transe(triples[:11], entities, rels)

def get_full_embeddings():
    return train_transe(triples, all_entities, all_rels)

if __name__ == "__main__":
    ent_emb, rel_emb, ent2id, rel2id = get_embeddings()

    # Test: h + r should land closest to correct tail
    h_vec = ent_emb[ent2id["Einstein"]] + rel_emb[rel2id["won"]]
    for e in ["Nobel_Physics", "Nobel_Chemistry", "Warsaw"]:
        d = np.linalg.norm(h_vec - ent_emb[ent2id[e]])
        print(f"  d(Einstein+won, {e}) = {d:.3f}")
    # Nobel_Physics should have smallest distance
