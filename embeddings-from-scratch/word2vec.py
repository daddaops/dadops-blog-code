"""
Word2Vec Skip-Gram with Negative Sampling — From Scratch

Implements the full Word2Vec pipeline:
1. Vocabulary building and frequency counting
2. Training pair generation (skip-gram with context window)
3. Negative sampling with frequency^0.75 distribution
4. Gradient descent training with sigmoid loss
5. Cosine similarity nearest-neighbor queries

Blog post: https://dadops.dev/blog/embeddings-from-scratch/
"""
import numpy as np


# --- Block 1: One-hot encoding demo ---
def demo_one_hot():
    vocab = ["cat", "dog", "fish", "bird", "king"]
    print("One-hot encoding:")
    for word in vocab:
        vec = np.zeros(len(vocab))
        vec[vocab.index(word)] = 1.0
        print(f"  {word:>5s} → {vec}")
    print()


# --- Block 2: Training pair generation ---
def generate_training_pairs(sentences, window=2):
    """For each center word, pair it with every context word in the window."""
    pairs = []
    for sentence in sentences:
        words = sentence.lower().split()
        for i, center in enumerate(words):
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((center, words[j]))
    return pairs


# --- Block 4: Noise distribution ---
def build_noise_distribution(word_counts):
    """Frequency^(3/4) — balances rare and common words."""
    counts = np.array(list(word_counts.values()), dtype=np.float64)
    powered = counts ** 0.75
    return powered / powered.sum()


def get_negative_samples(noise_dist, k=5, exclude=None):
    """Sample k words that probably aren't real context words."""
    negatives = []
    while len(negatives) < k:
        idx = np.random.choice(len(noise_dist), p=noise_dist)
        if idx != exclude:
            negatives.append(idx)
    return negatives


# --- Block 6: Complete Word2Vec class ---
class Word2Vec:
    def __init__(self, sentences, dim=20, window=2, neg_k=5, lr=0.05):
        self.word2idx, self.idx2word, counts = {}, [], {}
        for s in sentences:
            for w in s.lower().split():
                counts[w] = counts.get(w, 0) + 1
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.idx2word)
                    self.idx2word.append(w)

        V = len(self.idx2word)
        self.W_center = np.random.randn(V, dim) * 0.1
        self.W_context = np.random.randn(V, dim) * 0.1

        freq = np.array([counts[w] for w in self.idx2word], dtype=np.float64)
        self.noise = freq ** 0.75
        self.noise /= self.noise.sum()

        self.pairs = []
        for s in sentences:
            ids = [self.word2idx[w] for w in s.lower().split()]
            for i, c in enumerate(ids):
                for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                    if i != j:
                        self.pairs.append((c, ids[j]))

        self.lr, self.neg_k = lr, neg_k

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def train(self, epochs=1000):
        losses = []
        for epoch in range(epochs):
            np.random.shuffle(self.pairs)
            total = 0.0
            for ci, co in self.pairs:
                vc = self.W_center[ci].copy()
                # Positive
                s = self._sigmoid(np.dot(vc, self.W_context[co]))
                total += -np.log(s + 1e-10)
                grad = (s - 1) * self.W_context[co]
                self.W_context[co] -= self.lr * (s - 1) * vc
                # Negatives
                negs = np.random.choice(len(self.idx2word), self.neg_k, p=self.noise)
                for ni in negs:
                    s = self._sigmoid(np.dot(vc, self.W_context[ni]))
                    total += -np.log(1 - s + 1e-10)
                    grad += s * self.W_context[ni]
                    self.W_context[ni] -= self.lr * s * vc
                self.W_center[ci] -= self.lr * grad

            losses.append(total / len(self.pairs))
            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | Loss: {losses[-1]:.4f}")
        return losses

    def most_similar(self, word, k=5):
        v = self.W_center[self.word2idx[word]]
        norms = np.linalg.norm(self.W_center, axis=1)
        sims = (self.W_center @ v) / (norms * np.linalg.norm(v) + 1e-10)
        top = np.argsort(-sims)[1:k + 1]
        return [(self.idx2word[i], f"{sims[i]:.3f}") for i in top]


# --- Block 10: Analogy ---
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def analogy(model, a, b, c, k=3):
    """a is to b as c is to ???"""
    va = model.W_center[model.word2idx[a]]
    vb = model.W_center[model.word2idx[b]]
    vc = model.W_center[model.word2idx[c]]

    target = vb - va + vc

    exclude = {a, b, c}
    results = []
    for word, idx in model.word2idx.items():
        if word not in exclude:
            sim = cosine_sim(target, model.W_center[idx])
            results.append((word, sim))

    results.sort(key=lambda x: -x[1])
    return results[:k]


if __name__ == "__main__":
    # Demo one-hot encoding
    demo_one_hot()

    # Demo training pairs
    print("Training pairs from 'the cat sat on the mat':")
    pairs = generate_training_pairs(["the cat sat on the mat"])
    for center, context in pairs[:8]:
        print(f"  center={center:>5s}  →  context={context}")
    print()

    # Train Word2Vec on toy corpus
    corpus = [
        "the king ruled the kingdom",
        "the queen ruled the kingdom",
        "king and queen are royalty",
        "the man and woman walked together",
        "a boy and girl played together",
        "the cat sat on the mat",
        "the dog ran in the park",
        "cat and dog are pets",
        "a bird flew over the tree",
        "fish swam in the river",
        "bird and fish are animals",
        "the river flows past the tree",
        "sun shone over the park",
        "bread and cheese for lunch",
        "rice and soup for dinner",
        "the king ate bread and cheese",
        "the queen ate rice and soup",
        "the boy played with the dog",
        "the girl played with the cat",
        "the man walked in the park",
        "the woman walked by the river",
    ]

    np.random.seed(42)
    model = Word2Vec(corpus, dim=20, window=2, neg_k=5, lr=0.05)
    losses = model.train(epochs=1000)

    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"\nMost similar to 'king':  {model.most_similar('king')}")
    print(f"Most similar to 'cat':   {model.most_similar('cat')}")
    print(f"Most similar to 'bread': {model.most_similar('bread')}")

    # Analogy test
    print(f"\nking is to queen as man is to ???")
    print(f"  {analogy(model, 'king', 'queen', 'man')}")
    print(f"\ncat is to dog as boy is to ???")
    print(f"  {analogy(model, 'cat', 'dog', 'boy')}")
