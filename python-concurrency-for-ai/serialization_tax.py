"""
Serialization tax: pickle vs shared memory for NumPy arrays.

Benchmarks pickle round-trip vs shared memory for arrays of
various sizes (1 MB to 500 MB), demonstrating the crossover point
where shared memory becomes dramatically faster.

From: https://dadops.dev/blog/python-concurrency-for-ai/
"""
import numpy as np
import pickle
import time
from multiprocessing import shared_memory

def benchmark_pickle(arr):
    """Measure round-trip pickle time for a NumPy array."""
    t0 = time.perf_counter()
    data = pickle.dumps(arr, protocol=5)
    t_ser = time.perf_counter() - t0

    t0 = time.perf_counter()
    pickle.loads(data)
    t_deser = time.perf_counter() - t0
    return t_ser, t_deser

def benchmark_shared_memory(arr):
    """Measure shared memory setup + attach time."""
    t0 = time.perf_counter()
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype,
                            buffer=shm.buf)
    np.copyto(shared_arr, arr)
    t_create = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Simulate worker attaching to existing shared memory
    shm2 = shared_memory.SharedMemory(name=shm.name)
    worker_arr = np.ndarray(arr.shape, dtype=arr.dtype,
                            buffer=shm2.buf)
    _ = worker_arr.sum()  # force read
    t_attach = time.perf_counter() - t0

    shm2.close()
    shm.close()
    shm.unlink()
    return t_create, t_attach

if __name__ == "__main__":
    for size_mb in [1, 10, 50, 100, 500]:
        n = size_mb * 1024 * 1024 // 8  # float64 = 8 bytes
        arr = np.random.randn(n)

        t_ser, t_deser = benchmark_pickle(arr)
        t_create, t_attach = benchmark_shared_memory(arr)

        print(f"{size_mb:>4d} MB | Pickle: {(t_ser+t_deser)*1000:>7.1f}ms | "
              f"SharedMem: {(t_create+t_attach)*1000:>7.1f}ms | "
              f"Speedup: {(t_ser+t_deser)/(t_create+t_attach):>5.1f}x")
