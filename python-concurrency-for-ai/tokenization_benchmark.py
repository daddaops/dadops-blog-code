"""
Parallel tokenization benchmark: sequential vs threading vs multiprocessing.

Benchmarks CPU-bound text tokenization at various worker counts to show
that threading can't help with CPU work (GIL) but multiprocessing can.

From: https://dadops.dev/blog/python-concurrency-for-ai/
"""
import time
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

VOCAB = {w: i for i, w in enumerate(
    "the a an is was were been be have has had do does "
    "did will would shall should may might can could of "
    "in to for on with at by from as into about between".split()
)}

def tokenize_batch(texts):
    """Tokenize a batch: lowercase, split, map to vocab IDs."""
    results = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        ids = [VOCAB.get(w, len(VOCAB)) for w in words]
        results.append(ids)
    return results

def make_texts(n):
    """Generate n synthetic text snippets (~20 words each)."""
    base = ("The quick model was trained on a large dataset "
            "of text from the internet with careful filtering")
    return [f"{base} sample {i}" for i in range(n)]

def benchmark(executor_cls, texts, n_workers, chunk_size=2500):
    chunks = [texts[i:i+chunk_size]
              for i in range(0, len(texts), chunk_size)]
    start = time.perf_counter()
    with executor_cls(max_workers=n_workers) as pool:
        list(pool.map(tokenize_batch, chunks))
    return time.perf_counter() - start

if __name__ == "__main__":
    texts = make_texts(100_000)

    # Sequential baseline
    t0 = time.perf_counter()
    tokenize_batch(texts)
    t_seq = time.perf_counter() - t0
    print(f"Sequential: {t_seq:.2f}s\n")

    for n_workers in [1, 2, 4, 8]:
        t_thr = benchmark(ThreadPoolExecutor, texts, n_workers)
        t_mp = benchmark(ProcessPoolExecutor, texts, n_workers)
        print(f"Workers={n_workers} | Threads: {t_thr:.2f}s "
              f"({t_seq/t_thr:.1f}x) | Processes: {t_mp:.2f}s "
              f"({t_seq/t_mp:.1f}x)")
