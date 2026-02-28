"""Benchmark Results Table — All Three Strategies Compared

Pretty-prints a comparison table of distributed training strategies
across four model sizes. Data is from reference benchmarks on 8×A100
(80 GB) nodes with NVLink.

Blog post: https://dadops.co/blog/distributed-training-benchmarks/
Code Block 5 from the blog.
"""

import dataclasses


@dataclasses.dataclass
class BenchmarkResult:
    model: str
    params: str
    strategy: str
    gpus: int
    throughput: str
    mem_per_gpu: str
    scaling_eff: str
    comm_overhead: str


results = [
    # --- ResNet-50 (25M params) — DDP territory ---
    BenchmarkResult("ResNet-50", "25M", "DDP",    1, "620 img/s",  "5.8 GB",  "1.00", "—"),
    BenchmarkResult("ResNet-50", "25M", "DDP",    4, "2,280 img/s","5.8 GB",  "0.92", "~8%"),
    BenchmarkResult("ResNet-50", "25M", "DDP",    8, "4,340 img/s","5.8 GB",  "0.87", "~13%"),
    BenchmarkResult("ResNet-50", "25M", "TP-2",   2, "510 img/s",  "3.6 GB",  "0.41", "~59%"),

    # --- GPT-2 (125M params) — DDP still leads ---
    BenchmarkResult("GPT-2",    "125M","DDP",     1, "185 tok/ms", "8.2 GB",  "1.00", "—"),
    BenchmarkResult("GPT-2",    "125M","DDP",     4, "665 tok/ms", "8.2 GB",  "0.90", "~10%"),
    BenchmarkResult("GPT-2",    "125M","DDP",     8, "1,260 tok/ms","8.2 GB", "0.85", "~15%"),
    BenchmarkResult("GPT-2",    "125M","TP-2",    2, "165 tok/ms", "5.1 GB",  "0.45", "~55%"),
    BenchmarkResult("GPT-2",    "125M","PP-2",    2, "160 tok/ms", "5.4 GB",  "0.43", "~11% bubble"),

    # --- 1.3B Transformer — DDP OOMs, sharding wins ---
    BenchmarkResult("LLM-1.3B", "1.3B","DDP",    1, "OOM",        ">80 GB",  "—",    "—"),
    BenchmarkResult("LLM-1.3B", "1.3B","FSDP2",  4, "82 tok/ms",  "24.1 GB", "—",    "~20%"),
    BenchmarkResult("LLM-1.3B", "1.3B","FSDP2",  8, "148 tok/ms", "14.6 GB", "0.90*","~22%"),
    BenchmarkResult("LLM-1.3B", "1.3B","TP-4",   4, "95 tok/ms",  "18.3 GB", "—",    "~35%"),
    BenchmarkResult("LLM-1.3B", "1.3B","PP-4",   4, "70 tok/ms",  "22.5 GB", "—",    "~19% bubble"),

    # --- 7B Transformer — requires 3D parallelism ---
    BenchmarkResult("LLM-7B",   "7B",  "DDP",    1, "OOM",        ">80 GB",  "—",    "—"),
    BenchmarkResult("LLM-7B",   "7B",  "FSDP2",  8, "38 tok/ms",  "32.4 GB", "—",    "~25%"),
    BenchmarkResult("LLM-7B",   "7B",  "TP-4+PP-2",8,"44 tok/ms", "18.7 GB", "—",    "~30% total"),
    BenchmarkResult("LLM-7B",   "7B",  "TP-4+FSDP",8,"46 tok/ms", "21.2 GB", "—",    "~28% total"),
]


if __name__ == "__main__":
    # Pretty-print the comparison table
    print(f"{'Model':<12} {'Params':<7} {'Strategy':<12} {'GPUs':>4} "
          f"{'Throughput':<14} {'Mem/GPU':<9} {'Scale':<6} {'Comm'}")
    print("=" * 85)

    current_model = ""
    for r in results:
        if r.model != current_model:
            if current_model:
                print("-" * 85)
            current_model = r.model
        print(f"{r.model:<12} {r.params:<7} {r.strategy:<12} {r.gpus:>4} "
              f"{r.throughput:<14} {r.mem_per_gpu:<9} "
              f"{r.scaling_eff:<6} {r.comm_overhead}")
