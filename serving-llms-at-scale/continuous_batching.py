"""
Continuous Batching Scheduler — iteration-level scheduling for LLM serving.

Blog post: https://dadops.dev/blog/serving-llms-at-scale/
Code Block 2 from "Serving LLMs at Scale: From Naive to vLLM"
"""
import random

def simulate_continuous_batching(num_requests, batch_size, min_tokens, max_tokens, seed=42):
    """Simulate iteration-level scheduling with a request queue."""
    rng = random.Random(seed)
    output_lengths = [rng.randint(min_tokens, max_tokens) for _ in range(num_requests)]

    queue = list(range(num_requests))  # Request IDs waiting
    active = {}   # {request_id: remaining_tokens}
    completed = 0
    total_steps = 0
    latencies = {}  # request_id -> (arrival_step, completion_step)
    arrival_step = {i: 0 for i in range(num_requests)}  # All arrive at step 0

    # Fill initial batch
    for _ in range(min(batch_size, len(queue))):
        req_id = queue.pop(0)
        active[req_id] = output_lengths[req_id]

    while active or queue:
        total_steps += 1

        # One decode step: every active request generates one token
        finished_ids = []
        for req_id in list(active):
            active[req_id] -= 1
            if active[req_id] <= 0:
                finished_ids.append(req_id)

        # Remove finished, add from queue
        for req_id in finished_ids:
            del active[req_id]
            latencies[req_id] = total_steps - arrival_step[req_id]
            completed += 1

        # Fill empty slots from queue
        while len(active) < batch_size and queue:
            req_id = queue.pop(0)
            active[req_id] = output_lengths[req_id]
            arrival_step[req_id] = total_steps

    tokens_generated = sum(output_lengths)
    throughput = tokens_generated / total_steps if total_steps > 0 else 0
    avg_latency = sum(latencies.values()) / len(latencies) if latencies else 0

    print(f"Continuous Batching (batch_size={batch_size})")
    print(f"  Requests: {num_requests}")
    print(f"  Total decode steps: {total_steps}")
    print(f"  Tokens generated:   {tokens_generated:,}")
    print(f"  Throughput:   {throughput:.1f} tokens/step")
    print(f"  Avg latency:  {avg_latency:.0f} steps/request")
    print(f"  Completed:    {completed}/{num_requests}")
    return throughput


if __name__ == "__main__":
    print("=== Variable output lengths (20-500 tokens) ===")
    tp_continuous = simulate_continuous_batching(100, 8, 20, 500)
    print()
    print("=== Narrow output lengths (200-300 tokens) ===")
    simulate_continuous_batching(100, 8, 200, 300)
