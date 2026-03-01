"""
GIL benchmark: threading vs multiprocessing for CPU and I/O tasks.

Shows that threading gives near-perfect speedup for I/O-bound work
but no speedup (or slowdown) for CPU-bound work due to the GIL.
Multiprocessing delivers true CPU parallelism.

From: https://dadops.dev/blog/python-concurrency-for-ai/
"""
import time
import threading
import multiprocessing
import sys

def cpu_task(n=25_000_000):
    """Pure CPU work: sum of squares."""
    total = 0
    for i in range(n):
        total += i * i
    return total

def io_task(duration=0.5):
    """Simulated I/O: network call or DB query."""
    time.sleep(duration)

def run_threaded(func, n_workers=4, **kwargs):
    threads = [threading.Thread(target=func, kwargs=kwargs)
               for _ in range(n_workers)]
    start = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    return time.perf_counter() - start

def run_multiprocess(func, n_workers=4, **kwargs):
    procs = [multiprocessing.Process(target=func, kwargs=kwargs)
             for _ in range(n_workers)]
    start = time.perf_counter()
    for p in procs: p.start()
    for p in procs: p.join()
    return time.perf_counter() - start

if __name__ == "__main__":
    n_workers = 4
    gil_status = "disabled" if hasattr(sys, "_is_gil_enabled") \
                 and not sys._is_gil_enabled() else "enabled"
    print(f"GIL: {gil_status} | Workers: {n_workers}\n")

    # I/O-bound benchmark
    t_seq = time.perf_counter()
    for _ in range(n_workers): io_task()
    t_seq = time.perf_counter() - t_seq
    t_thr = run_threaded(io_task, n_workers)
    print(f"I/O-bound  | Sequential: {t_seq:.2f}s | "
          f"Threaded: {t_thr:.2f}s | "
          f"Speedup: {t_seq/t_thr:.1f}x")

    # CPU-bound benchmark
    t_seq = time.perf_counter()
    for _ in range(n_workers): cpu_task()
    t_seq = time.perf_counter() - t_seq
    t_thr = run_threaded(cpu_task, n_workers)
    t_mp = run_multiprocess(cpu_task, n_workers)
    print(f"CPU-bound  | Sequential: {t_seq:.2f}s | "
          f"Threaded: {t_thr:.2f}s | "
          f"Multiproc: {t_mp:.2f}s | "
          f"MP speedup: {t_seq/t_mp:.1f}x")
