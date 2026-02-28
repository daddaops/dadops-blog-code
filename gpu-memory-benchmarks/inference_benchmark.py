"""Inference Memory Benchmarks — Memory by Precision and Batch Size

Measures inference memory for models at different precisions (FP32,
FP16, BF16, INT8, INT4) and batch sizes.

REQUIRES: NVIDIA GPU + HuggingFace transformers + bitsandbytes

Blog post: https://dadops.co/blog/gpu-memory-benchmarks/
Code Block 2 from the blog.
"""

import torch
from transformers import AutoModelForCausalLM


def benchmark_inference_memory(model_name, precisions, batch_sizes, seq_len=512):
    """Measure inference memory across precisions and batch sizes."""
    results = []

    for precision in precisions:
        dtype_map = {
            "FP32": torch.float32,
            "FP16": torch.float16,
            "BF16": torch.bfloat16,
        }
        load_kwargs = {}
        if precision in dtype_map:
            load_kwargs["torch_dtype"] = dtype_map[precision]
        elif precision == "INT8":
            load_kwargs["load_in_8bit"] = True
        elif precision == "INT4":
            load_kwargs["load_in_4bit"] = True

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", **load_kwargs
        )
        model_mem = torch.cuda.memory_allocated() / 1e9

        for bs in batch_sizes:
            torch.cuda.reset_peak_memory_stats()
            input_ids = torch.randint(0, 1000, (bs, seq_len), device="cuda")

            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=1)

            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            results.append({
                "precision": precision,
                "batch_size": bs,
                "model_gb": round(model_mem, 2),
                "peak_gb": round(peak_mem, 2),
                "kv_cache_gb": round(peak_mem - model_mem, 2),
            })
        del model

    return results


if __name__ == "__main__":
    print("This script requires an NVIDIA GPU with CUDA + HuggingFace transformers.")
    print("Usage: benchmark_inference_memory('model_name', ['FP16'], [1, 8])")
