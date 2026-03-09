"""BERT-style Masked Language Modeling."""
import numpy as np

# Build vocabulary from a small corpus
corpus = [
    "the cat sat on the mat",
    "the dog played in the park",
    "a bird flew over the tree",
    "the fish swam in the pond"
]
words = sorted(set(w for s in corpus for w in s.split()))
word2idx = {w: i + 3 for i, w in enumerate(words)}  # 0=PAD, 1=MASK, 2=UNK
idx2word = {i: w for w, i in word2idx.items()}
idx2word[1] = "[MASK]"
vocab_size = len(word2idx) + 3

# Tokenize
sentences = [[word2idx[w] for w in s.split()] for s in corpus]

# BERT-style masking: 15% of tokens, with 80/10/10 strategy
def mask_tokens(tokens, mask_prob=0.15):
    masked = tokens.copy()
    labels = [-1] * len(tokens)  # -1 means "ignore in loss"
    for i in range(len(tokens)):
        if np.random.random() < mask_prob:
            labels[i] = tokens[i]       # remember true token
            r = np.random.random()
            if r < 0.8:
                masked[i] = 1           # 80%: replace with [MASK]
            elif r < 0.9:
                masked[i] = np.random.randint(3, vocab_size)  # 10%: random
            # remaining 10%: keep original (no change)
    return masked, labels

# Simple encoder: embedding lookup + linear prediction head
embed_dim = 16
np.random.seed(42)
W_embed = np.random.randn(vocab_size, embed_dim) * 0.1
W_predict = np.random.randn(embed_dim, vocab_size) * 0.1

# Forward pass on "the cat sat on the mat"
sentence = sentences[0]
masked, labels = mask_tokens(sentence, mask_prob=0.5)  # high rate for demo

embeddings = W_embed[masked]          # (seq_len, embed_dim)
logits = embeddings @ W_predict       # (seq_len, vocab_size)

# Cross-entropy loss ONLY on masked positions
loss = 0.0
n_masked = 0
for i, label in enumerate(labels):
    if label != -1:
        exp_logits = np.exp(logits[i] - logits[i].max())
        probs = exp_logits / exp_logits.sum()
        loss -= np.log(probs[label] + 1e-10)
        n_masked += 1
        predicted = idx2word.get(np.argmax(logits[i]), "?")
        actual = idx2word.get(label, "?")
        print(f"Position {i}: true='{actual}', predicted='{predicted}'")

if n_masked:
    print(f"\nAvg loss over {n_masked} masked tokens: {loss / n_masked:.3f}")
    print(f"Random baseline loss: {np.log(vocab_size):.3f}")
