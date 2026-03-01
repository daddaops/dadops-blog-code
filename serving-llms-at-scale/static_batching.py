"""
Static Batching Simulator — measures padding waste from fixed-batch LLM serving.

Blog post: https://dadops.dev/blog/serving-llms-at-scale/
Code Block 1 from "Serving LLMs at Scale: From Naive to vLLM"
"""
import random
import statistics

def simulate_static_batching(num_requests, batch_size, min_tokens, max_tokens, seed=42):
    """Simulate a static batching LLM server and measure waste."""
    rng = random.Random(seed)
    output_lengths = [rng.randint(min_tokens, max_tokens) for _ in range(num_requests)]

    total_useful_steps = 0
    total_padded_steps = 0
    request_latencies = []

    for batch_start in range(0, num_requests, batch_size):
        batch = output_lengths[batch_start:batch_start + batch_size]
        max_len = max(batch)

        for length in batch:
            total_useful_steps += length
            total_padded_steps += max_len - length
            # Every request in the batch waits for the longest one
            request_latencies.append(max_len)

    waste_pct = total_padded_steps / (total_useful_steps + total_padded_steps) * 100
    throughput = total_useful_steps / sum(request_latencies) * batch_size

    print(f"Static Batching (batch_size={batch_size})")
    print(f"  Requests: {num_requests}")
    print(f"  Output lengths: {min_tokens}-{max_tokens} tokens")
    print(f"  Useful tokens:  {total_useful_steps:,}")
    print(f"  Padded tokens:  {total_padded_steps:,}")
    print(f"  Padding waste:  {waste_pct:.1f}%")
    print(f"  Avg latency:    {statistics.mean(request_latencies):.0f} steps/request")
    print(f"  Relative throughput: {throughput:.2f} tokens/step")


if __name__ == "__main__":
    # Wide variance in output length = lots of waste
    simulate_static_batching(100, batch_size=8, min_tokens=20, max_tokens=500)
    print()
    # Narrow variance = less waste, but still blocked
    simulate_static_batching(100, batch_size=8, min_tokens=200, max_tokens=300)
