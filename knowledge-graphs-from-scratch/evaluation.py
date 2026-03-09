import numpy as np

def evaluate_link_prediction(ent_emb, rel_emb, ent2id, rel2id,
                              test_triples, all_true):
    """Compute MR, MRR, Hits@1/3/10 with filtered ranking."""
    ranks = []
    n_ent = len(ent2id)
    id2ent = {v: k for k, v in ent2id.items()}

    for h, r, t in test_triples:
        hi, ri, ti = ent2id[h], rel2id[r], ent2id[t]
        # Score all entities as possible tails (TransE)
        scores = np.zeros(n_ent)
        for ei in range(n_ent):
            scores[ei] = -np.linalg.norm(
                ent_emb[hi] + rel_emb[ri] - ent_emb[ei])

        # Filter: mask other true tails
        for ei in range(n_ent):
            if ei != ti and (h, r, id2ent[ei]) in all_true:
                scores[ei] = -1e9

        rank = 1 + np.sum(scores > scores[ti])
        ranks.append(rank)

    ranks = np.array(ranks)
    print(f"MR: {np.mean(ranks):.1f} | MRR: {np.mean(1.0/ranks):.3f} | "
          f"H@1: {np.mean(ranks <= 1):.2f} | "
          f"H@3: {np.mean(ranks <= 3):.2f} | "
          f"H@10: {np.mean(ranks <= 10):.2f}")
