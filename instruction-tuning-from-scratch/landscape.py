import numpy as np

# The key papers and what they proved
papers = [
    {"year": 2022, "name": "InstructGPT", "authors": "Ouyang et al.",
     "finding": "1.3B tuned model beats 175B base model",
     "insight": "Alignment is more cost-effective than scale",
     "sft_examples": 13000},
    {"year": 2023, "name": "Self-Instruct", "authors": "Wang et al.",
     "finding": "LLMs can generate their own training data",
     "insight": "Instruction diversity matters more than human curation",
     "sft_examples": 52000},
    {"year": 2023, "name": "Alpaca", "authors": "Taori et al.",
     "finding": "7B model matches much larger model behaviors",
     "insight": "Instruction tuning is accessible to everyone",
     "sft_examples": 52000},
    {"year": 2023, "name": "LIMA", "authors": "Zhou et al.",
     "finding": "1,000 examples beat 52,000 examples",
     "insight": "Data quality is everything",
     "sft_examples": 1000},
]

# Data scaling experiment: quality vs dataset size
dataset_sizes = [100, 500, 1000, 5000, 10000, 50000]
# Quality follows a logarithmic curve (LIMA finding)
quality_scores = [0.45, 0.68, 0.82, 0.87, 0.89, 0.90]

print("=== Landmark Papers ===")
for p in papers:
    print(f"\n{p['year']} - {p['name']} ({p['authors']})")
    print(f"  Finding: {p['finding']}")
    print(f"  Insight: {p['insight']}")
    print(f"  SFT examples: {p['sft_examples']:,}")

print("\n\n=== Data Scaling: Quality vs Dataset Size ===")
print(f"{'Examples':>10} {'Quality':>10} {'Marginal gain':>15}")
print("-" * 38)
for i, (n, q) in enumerate(zip(dataset_sizes, quality_scores)):
    prev_q = quality_scores[i - 1] if i > 0 else 0
    gain = q - prev_q
    bar = "#" * int(q * 30)
    print(f"{n:>10,} {q:>9.0%} {'+' + f'{gain:.0%}':>14}  {bar}")
# The massive jump is 100 -> 1000 examples (+37%)
# After 1000, gains are marginal: 1000 -> 50000 gives only +8%
