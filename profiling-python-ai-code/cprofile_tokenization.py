"""
cProfile on a batch tokenization pipeline.

Profiles simulated BPE tokenization of 5,000 documents using cProfile
and prints the top functions by cumulative time.

From: https://dadops.dev/blog/profiling-python-ai-code/
"""
import cProfile
import pstats
import io

def tokenize_documents(texts, vocab_size=30522):
    """Tokenize a batch of documents with simulated BPE."""
    tokenized = []
    for text in texts:
        words = text.lower().split()
        tokens = []
        for word in words:
            token_id = hash(word) % vocab_size
            tokens.append(token_id)
            # Simulate subword splits for longer words
            if len(word) > 6:
                for i in range(0, len(word) - 3, 3):
                    tokens.append(hash(word[i:i+3]) % vocab_size)
        tokenized.append(tokens)
    return tokenized

if __name__ == "__main__":
    # Generate 5,000 synthetic documents
    docs = [
        f"Document {i} covers machine learning topics "
        f"including transformers attention embeddings " * 20
        for i in range(5000)
    ]

    # Profile the tokenization
    profiler = cProfile.Profile()
    profiler.enable()
    result = tokenize_documents(docs)
    profiler.disable()

    # Analyze — sort by cumulative time
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(10)
    print(stream.getvalue())
