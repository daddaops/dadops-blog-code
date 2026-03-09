import numpy as np
from transe import get_full_embeddings

def multi_hop_query(ent_emb, rel_emb, ent2id, rel2id,
                    start, relation_path):
    """Answer multi-hop queries by composing TransE translations."""
    id2ent = {v: k for k, v in ent2id.items()}

    # Compose: start entity + sum of relation vectors
    composed = ent_emb[ent2id[start]].copy()
    for rel in relation_path:
        composed += rel_emb[rel2id[rel]]

    # Rank all entities by distance
    distances = np.linalg.norm(ent_emb - composed, axis=1)
    ranked = np.argsort(distances)

    path_str = " -> ".join(relation_path)
    print(f"Query: {start} -> {path_str} -> ?")
    for i in range(3):
        eid = ranked[i]
        print(f"  #{i+1}: {id2ent[eid]} (d={distances[eid]:.3f})")

ent_emb, rel_emb, ent2id, rel2id = get_full_embeddings()

# "What country was Einstein born in?" = born_in + located_in
multi_hop_query(ent_emb, rel_emb, ent2id, rel2id,
                "Einstein", ["born_in", "located_in"])
# Should rank 'Germany' highest
