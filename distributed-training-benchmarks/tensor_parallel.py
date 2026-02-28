"""Column-Parallel and Row-Parallel Linear Layers

Educational implementation of the two fundamental tensor parallelism
patterns from Megatron-LM: column-parallel and row-parallel linear
layers.

REQUIRES: Multiple NVIDIA GPUs with CUDA support and torch.distributed.

Blog post: https://dadops.co/blog/distributed-training-benchmarks/
Code Block 2 from the blog.
"""

import torch
import torch.distributed as dist


def column_parallel_linear(x, weight_full, world_size, rank):
    """Column-parallel: split weight columns, all-gather output."""
    # weight_full: [in_features, out_features]
    chunk_size = weight_full.size(1) // world_size
    weight_local = weight_full[:, rank * chunk_size:(rank + 1) * chunk_size]

    # Each GPU computes its slice: [batch, in] @ [in, out/P] = [batch, out/P]
    y_local = x @ weight_local

    # All-gather: concatenate partial outputs along feature dim
    gathered = [torch.empty_like(y_local) for _ in range(world_size)]
    dist.all_gather(gathered, y_local)
    return torch.cat(gathered, dim=-1)  # [batch, out]


def row_parallel_linear(x_splits, weight_full, world_size, rank):
    """Row-parallel: split weight rows and input, all-reduce output."""
    # weight_full: [in_features, out_features]
    chunk_size = weight_full.size(0) // world_size
    weight_local = weight_full[rank * chunk_size:(rank + 1) * chunk_size, :]

    # Each GPU: [batch, in/P] @ [in/P, out] = [batch, out]
    y_local = x_splits[rank] @ weight_local

    # All-reduce: sum partial outputs
    dist.all_reduce(y_local, op=dist.ReduceOp.SUM)
    return y_local  # [batch, out]


# --- Modern PyTorch native approach (3 lines!) ---
# from torch.distributed.tensor.parallel import (
#     ColwiseParallel, RowwiseParallel, parallelize_module
# )
# from torch.distributed.device_mesh import init_device_mesh
#
# mesh = init_device_mesh("cuda", (world_size,))
# parallelize_module(model.attention, mesh, {
#     "qkv_proj": ColwiseParallel(),
#     "out_proj": RowwiseParallel(),
# })
# parallelize_module(model.ffn, mesh, {
#     "fc1": ColwiseParallel(),
#     "fc2": RowwiseParallel(),
# })


if __name__ == "__main__":
    print("This script requires multi-GPU torch.distributed setup.")
    print("It demonstrates column-parallel and row-parallel linear layer patterns.")
    print("Run in a distributed context with torchrun.")
