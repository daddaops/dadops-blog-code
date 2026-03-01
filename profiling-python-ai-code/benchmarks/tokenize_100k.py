"""Batch tokenization workload: simulated BPE on 100K short docs."""
import sys

def tokenize_documents(texts, vocab_size=30522):
    tokenized = []
    for text in texts:
        words = text.lower().split()
        tokens = []
        for word in words:
            token_id = hash(word) % vocab_size
            tokens.append(token_id)
            if len(word) > 6:
                for i in range(0, len(word) - 3, 3):
                    tokens.append(hash(word[i:i+3]) % vocab_size)
        tokenized.append(tokens)
    return tokenized

if __name__ == "__main__":
    docs = [f"Document {i} about transformers attention embeddings " * 5
            for i in range(100_000)]
    tokenize_documents(docs)
