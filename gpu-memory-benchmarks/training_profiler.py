"""Training Memory Stage Profiler — Waterfall Breakdown

Profiles training memory at each stage: model loading, forward pass,
backward pass, optimizer step, and cleanup.

REQUIRES: NVIDIA GPU with CUDA support + a loaded model.

Blog post: https://dadops.co/blog/gpu-memory-benchmarks/
Code Block 3 from the blog.
"""

import torch


def profile_training_stages(model, input_ids, labels):
    """Break down memory consumption at each training stage."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    stages = {}

    # Stage 1: Model already loaded — measure baseline
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    stages["model_params"] = torch.cuda.memory_allocated() / 1e9

    # Stage 2: Forward pass — activations accumulate
    torch.cuda.reset_peak_memory_stats()
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    torch.cuda.synchronize()
    stages["after_forward"] = torch.cuda.memory_allocated() / 1e9
    stages["forward_peak"] = torch.cuda.max_memory_allocated() / 1e9

    # Stage 3: Backward pass — gradients allocated
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    torch.cuda.synchronize()
    stages["after_backward"] = torch.cuda.memory_allocated() / 1e9
    stages["backward_peak"] = torch.cuda.max_memory_allocated() / 1e9

    # Stage 4: Optimizer step — optimizer states created on first call
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()
    torch.cuda.synchronize()
    stages["after_optimizer"] = torch.cuda.memory_allocated() / 1e9
    stages["optimizer_peak"] = torch.cuda.max_memory_allocated() / 1e9

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    stages["after_cleanup"] = torch.cuda.memory_allocated() / 1e9

    print("Training Memory Waterfall (7B model, FP16, batch=4, seq=512):")
    print(f"  Model parameters:     {stages['model_params']:.1f} GB")
    print(f"  After forward pass:   {stages['after_forward']:.1f} GB  "
          f"(+{stages['after_forward'] - stages['model_params']:.1f} GB activations)")
    print(f"  After backward pass:  {stages['after_backward']:.1f} GB  "
          f"(+{stages['after_backward'] - stages['after_forward']:.1f} GB gradients)")
    print(f"  After optimizer step: {stages['after_optimizer']:.1f} GB  "
          f"(+{stages['after_optimizer'] - stages['after_backward']:.1f} GB opt states)")
    print(f"  After cleanup:        {stages['after_cleanup']:.1f} GB")
    print(f"  Peak during backward: {stages['backward_peak']:.1f} GB")
    return stages


if __name__ == "__main__":
    print("This script requires an NVIDIA GPU + a loaded model.")
    print("Usage: profile_training_stages(model, input_ids, labels)")
