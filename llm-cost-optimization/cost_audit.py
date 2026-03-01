"""
LLM Cost Audit — analyzes usage logs and generates optimization report.

From: https://dadops.dev/blog/llm-cost-optimization/
Code Block 7: "The Complete Optimization Stack — Putting It All Together"

Identifies model routing, caching, and prompt compression opportunities.
"""

from dataclasses import dataclass


@dataclass
class UsageRecord:
    feature: str
    model: str
    input_tokens: int
    output_tokens: int
    prompt_hash: str
    is_cacheable: bool = True


def audit_llm_costs(records: list[UsageRecord],
                    pricing: dict[str, tuple[float, float]]) -> str:
    """Analyze usage logs and generate optimization report."""
    total_cost = 0.0
    feature_costs: dict[str, float] = {}
    model_usage: dict[str, int] = {}
    duplicate_prompts: dict[str, int] = {}
    total_input = 0

    for rec in records:
        inp_rate, out_rate = pricing.get(rec.model, (3.0, 15.0))
        cost = (rec.input_tokens * inp_rate
                + rec.output_tokens * out_rate) / 1e6
        total_cost += cost
        feature_costs[rec.feature] = (
            feature_costs.get(rec.feature, 0) + cost)
        model_usage[rec.model] = model_usage.get(rec.model, 0) + 1
        total_input += rec.input_tokens
        if rec.is_cacheable:
            duplicate_prompts[rec.prompt_hash] = (
                duplicate_prompts.get(rec.prompt_hash, 0) + 1)

    # Find opportunities
    opportunities = []

    # 1. Model routing opportunity
    expensive_model_count = sum(
        ct for m, ct in model_usage.items()
        if m in ("opus", "gpt-4o"))
    if expensive_model_count > len(records) * 0.3:
        savings = total_cost * 0.35
        opportunities.append(
            f"Route {expensive_model_count} expensive-model requests "
            f"to cheaper tiers → ~${savings:.0f}/month savings")

    # 2. Caching opportunity
    dupes = sum(1 for c in duplicate_prompts.values() if c > 1)
    dupe_pct = dupes / len(records) * 100 if records else 0
    if dupe_pct > 10:
        savings = total_cost * (dupe_pct / 100) * 0.95
        opportunities.append(
            f"{dupe_pct:.0f}% duplicate prompts detected → "
            f"add caching for ~${savings:.0f}/month savings")

    # 3. Prompt compression opportunity
    avg_input = total_input / len(records) if records else 0
    if avg_input > 2000:
        savings = total_cost * 0.12
        opportunities.append(
            f"Avg prompt is {avg_input:.0f} tokens → "
            f"compress to ~{avg_input * 0.65:.0f} for "
            f"~${savings:.0f}/month savings")

    report = [f"=== LLM Cost Audit ({len(records)} requests) ===",
              f"Total monthly spend: ${total_cost:.0f}",
              "", "Top opportunities:"]
    for i, opp in enumerate(opportunities, 1):
        report.append(f"  {i}. {opp}")
    return "\n".join(report)


if __name__ == "__main__":
    import random
    import hashlib

    random.seed(42)

    # Simulate 30 days of realistic usage logs
    pricing = {
        "opus":   (15.00, 75.00),
        "sonnet": (3.00, 15.00),
        "haiku":  (1.00, 5.00),
    }

    # Create 1000 usage records mimicking a real workload:
    # - 40% opus (over-provisioned — should trigger routing recommendation)
    # - 35% sonnet, 25% haiku
    # - 20% duplicate prompts (should trigger caching recommendation)
    # - Average 3500 input tokens (should trigger compression recommendation)
    records = []
    prompt_pool = [f"prompt_{i}" for i in range(80)]  # 80 unique prompts

    for i in range(1000):
        # Model distribution
        r = random.random()
        if r < 0.40:
            model = "opus"
        elif r < 0.75:
            model = "sonnet"
        else:
            model = "haiku"

        # Feature distribution
        feature = random.choice(["chatbot", "doc-processor", "code-gen", "search"])

        # Token counts (average ~3500 input)
        input_tokens = random.randint(1500, 5500)
        output_tokens = random.randint(200, 800)

        # Prompt hash — reuse from pool to create duplicates
        if random.random() < 0.25:
            prompt = random.choice(prompt_pool[:20])  # high-repeat subset
        else:
            prompt = random.choice(prompt_pool)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        records.append(UsageRecord(
            feature=feature,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_hash=prompt_hash,
        ))

    report = audit_llm_costs(records, pricing)
    print(report)

    # Show breakdown
    print("\n=== Detailed Breakdown ===")
    model_counts = {}
    for rec in records:
        model_counts[rec.model] = model_counts.get(rec.model, 0) + 1
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count} requests ({count/len(records)*100:.0f}%)")

    # Verify duplicate detection
    from collections import Counter
    hash_counts = Counter(rec.prompt_hash for rec in records)
    dupes = sum(1 for c in hash_counts.values() if c > 1)
    print(f"\n  Unique prompt hashes: {len(hash_counts)}")
    print(f"  Hashes with duplicates: {dupes} ({dupes/len(records)*100:.0f}%)")
    avg_input = sum(r.input_tokens for r in records) / len(records)
    print(f"  Average input tokens: {avg_input:.0f}")
