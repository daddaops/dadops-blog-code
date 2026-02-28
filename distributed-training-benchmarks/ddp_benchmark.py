"""DDP Training Loop with Throughput Measurement

Benchmarks PyTorch DistributedDataParallel (DDP) on ResNet-50 with
synthetic data. Measures throughput (images/sec) and peak memory.

REQUIRES: Multiple NVIDIA GPUs with CUDA support.
Launch with: torchrun --nproc_per_node=N ddp_benchmark.py

Blog post: https://dadops.co/blog/distributed-training-benchmarks/
Code Block 1 from the blog.
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torchvision.models import resnet50


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def benchmark_ddp(epochs=3, batch_size=64, num_samples=4096):
    local_rank = setup()
    world_size = dist.get_world_size()

    model = resnet50(num_classes=1000).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Synthetic dataset — same size on every rank
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 1000, (num_samples,))
    dataset = TensorDataset(images, labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                  rank=dist.get_rank())
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Warm-up pass
    for batch_img, batch_lbl in loader:
        loss = criterion(model(batch_img.to(local_rank)),
                         batch_lbl.to(local_rank))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        break

    torch.cuda.synchronize()
    start = time.perf_counter()
    total_samples = 0

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch_img, batch_lbl in loader:
            out = model(batch_img.to(local_rank))
            loss = criterion(out, batch_lbl.to(local_rank))
            loss.backward()       # gradient sync happens here
            optimizer.step()
            optimizer.zero_grad()
            total_samples += batch_img.size(0) * world_size

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    throughput = total_samples / elapsed
    mem_gb = torch.cuda.max_memory_allocated(local_rank) / 1e9

    if local_rank == 0:
        print(f"GPUs: {world_size}  Throughput: {throughput:.0f} img/s  "
              f"Peak mem: {mem_gb:.1f} GB  Time: {elapsed:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    benchmark_ddp()
