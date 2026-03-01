"""
PagedAttention Memory Manager — simplified page-table KV cache management.

Blog post: https://dadops.dev/blog/serving-llms-at-scale/
Code Block 3 from "Serving LLMs at Scale: From Naive to vLLM"
"""
import random


class PagedKVCacheManager:
    """Simplified PagedAttention memory manager with page tables."""

    def __init__(self, total_pages, tokens_per_page=16):
        self.total_pages = total_pages
        self.tokens_per_page = tokens_per_page
        self.free_pages = list(range(total_pages))  # Free page pool
        self.page_tables = {}    # request_id -> [page_indices]
        self.ref_counts = [0] * total_pages  # For shared prefix pages

    def allocate_request(self, request_id, num_tokens):
        """Allocate pages for a new request. Pages grow on demand."""
        pages_needed = (num_tokens + self.tokens_per_page - 1) // self.tokens_per_page
        if pages_needed > len(self.free_pages):
            return False  # Out of memory
        allocated = [self.free_pages.pop() for _ in range(pages_needed)]
        for p in allocated:
            self.ref_counts[p] = 1
        self.page_tables[request_id] = allocated
        return True

    def grow_request(self, request_id, new_tokens):
        """Allocate additional pages as sequence grows."""
        current_pages = len(self.page_tables[request_id])
        current_capacity = current_pages * self.tokens_per_page
        total_tokens = current_capacity + new_tokens
        extra_pages = (total_tokens + self.tokens_per_page - 1) // self.tokens_per_page - current_pages
        if extra_pages > len(self.free_pages):
            return False
        for _ in range(extra_pages):
            page = self.free_pages.pop()
            self.ref_counts[page] = 1
            self.page_tables[request_id].append(page)
        return True

    def share_prefix(self, source_id, target_id, prefix_pages):
        """Share prefix pages between requests (copy-on-write)."""
        shared = self.page_tables[source_id][:prefix_pages]
        for p in shared:
            self.ref_counts[p] += 1
        self.page_tables[target_id] = shared.copy()

    def free_request(self, request_id):
        """Free all pages for a completed request."""
        for page in self.page_tables.pop(request_id, []):
            self.ref_counts[page] -= 1
            if self.ref_counts[page] == 0:
                self.free_pages.append(page)

    def utilization(self):
        allocated = self.total_pages - len(self.free_pages)
        return allocated / self.total_pages * 100


if __name__ == "__main__":
    # Demo: compare contiguous vs paged allocation
    total_gpu_pages = 256  # Represent 256 pages of KV cache memory

    manager = PagedKVCacheManager(total_gpu_pages, tokens_per_page=16)

    # Simulate 10 requests with variable lengths
    rng = random.Random(42)
    request_lengths = [rng.randint(32, 512) for _ in range(10)]

    print("PagedAttention Allocation:")
    for i, length in enumerate(request_lengths):
        success = manager.allocate_request(f"req_{i}", length)
        pages_used = len(manager.page_tables.get(f"req_{i}", []))
        print(f"  req_{i}: {length} tokens -> {pages_used} pages {'[OK]' if success else '[OOM]'}")

    print(f"\n  Memory utilization: {manager.utilization():.1f}%")
    print(f"  Free pages: {len(manager.free_pages)}/{total_gpu_pages}")

    # Now show prefix sharing
    print("\nPrefix Sharing Demo:")
    manager2 = PagedKVCacheManager(128, tokens_per_page=16)
    manager2.allocate_request("base", 256)  # 256-token system prompt = 16 pages
    base_pages = len(manager2.page_tables["base"])
    print(f"  Base prompt: 256 tokens = {base_pages} pages")

    for i in range(5):
        manager2.share_prefix("base", f"shared_{i}", base_pages)
        manager2.grow_request(f"shared_{i}", rng.randint(32, 128))
        total = len(manager2.page_tables[f"shared_{i}"])
        print(f"  shared_{i}: {base_pages} shared + {total - base_pages} unique = {total} total pages")

    print(f"\n  With sharing:    {manager2.utilization():.1f}% utilized")
    print(f"  Without sharing: would need {base_pages * 5} extra pages for duplicated prefixes")
