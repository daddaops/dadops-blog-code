import math
from collections import Counter

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [d.lower().split() for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        self.df = Counter()           # document frequency per term
        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1
        self.N = len(self.docs)

    def score(self, query, doc_idx):
        doc = self.docs[doc_idx]
        tf = Counter(doc)
        dl = len(doc)
        s = 0.0
        for term in query.lower().split():
            if term not in tf:
                continue
            idf = math.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
            numerator = tf[term] * (self.k1 + 1)
            denominator = tf[term] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * numerator / denominator
        return s

    def search(self, query, top_k=3):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        return sorted(scores, key=lambda x: -x[1])[:top_k]

# The vocabulary mismatch problem in action
corpus = [
    "Plumbing repair: stopping drips from bathroom fixtures",
    "How to fix a leaking kitchen faucet step by step",
    "Best bathroom renovation ideas for small spaces",
    "Emergency pipe burst: shutting off the main water valve",
    "Guide for replacing worn rubber washers in tap valves",
]

bm25 = BM25(corpus)
results = bm25.search("how to fix a leaking faucet")
for idx, score in results:
    print(f"  [{score:.2f}] {corpus[idx]}")
# [7.66] How to fix a leaking kitchen faucet step by step
# [0.00] Plumbing repair: stopping drips from bathroom fixtures  <-- zero!
# [0.00] Best bathroom renovation ideas for small spaces
