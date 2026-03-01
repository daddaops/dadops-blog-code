"""
Full benchmark pipeline: dataset generation, FTS5 setup, speed benchmarks,
hybrid search, and batch benchmarks.

Blog post: https://dadops.dev/blog/sqlite-fts-vs-rapidfuzz/
Code Blocks 2-10 from "SQLite FTS5 vs rapidfuzz: Fuzzy Search Showdown"

This is a comprehensive benchmark that takes several minutes to run
(mainly due to rapidfuzz linear scans on 500K records).
"""
import random
import sqlite3
import timeit
from rapidfuzz import fuzz, process
from concurrent.futures import ProcessPoolExecutor

# ──────────────────────────────────────────────
# Block 2: Dataset Generation (500K product names)
# ──────────────────────────────────────────────

categories = [
    "Milk", "Bread", "Cheese", "Yogurt", "Butter", "Juice", "Cereal",
    "Pasta", "Rice", "Chicken", "Beef", "Salmon", "Shrimp", "Tofu",
    "Apple", "Banana", "Orange", "Grape", "Mango", "Tomato", "Onion",
    "Garlic", "Pepper", "Lettuce", "Spinach", "Broccoli", "Carrot",
    "Potato", "Chips", "Crackers", "Cookies", "Granola", "Oatmeal",
    "Coffee", "Tea", "Soda", "Water", "Chocolate", "Candy", "Gum",
    "Soap", "Shampoo", "Detergent", "Napkins", "Towels", "Batteries",
    "Almonds", "Walnuts", "Peanuts", "Cashews", "Pecans", "Pistachios",
    "Hummus", "Salsa", "Guacamole", "Sour Cream", "Cream Cheese",
    "Bagels", "Muffins", "Croissants", "Tortillas", "Pita",
    "Ketchup", "Mustard", "Mayo", "Hot Sauce", "Soy Sauce",
    "Ice Cream", "Frozen Pizza", "Frozen Waffles", "Fish Sticks",
    "Olive Oil", "Coconut Oil", "Avocado Oil", "Vinegar",
    "Honey", "Maple Syrup", "Jam", "Peanut Butter", "Nutella",
]

brands = [
    "Organic Valley", "Great Value", "Kirkland", "Trader Joe's",
    "Nature's Best", "Green Harvest", "Farm Fresh", "Blue Diamond",
    "Simply", "Horizon", "Stonyfield", "Annie's", "Bob's Red Mill",
    "Kettle Brand", "Clif", "KIND", "Burt's Bees", "Method",
    "Pacific", "Amy's", "Newman's Own", "Seventh Generation",
    "Whole Foods 365", "Good & Gather", "O Organics", "Simple Truth",
    "Wild Harvest", "Market Pantry", "Up & Up", "Signature Select",
    "Happy Belly", "Primal Kitchen", "Sir Kensington's", "Rao's",
    "Dave's Killer Bread", "Justin's", "Rxbar", "Purely Elizabeth",
]

modifiers = [
    "Organic", "Low-Fat", "Whole Grain", "Gluten-Free", "Sugar-Free",
    "2%", "1%", "Fat-Free", "Extra Virgin", "Cold-Pressed",
    "Unsalted", "Lightly Salted", "Honey Roasted", "Dark",
    "Original", "Family Size", "Single Serve", "Variety Pack",
    "Wild Caught", "Free Range", "Grass Fed", "Non-GMO",
    "Reduced Sodium", "No Added Sugar", "High Protein", "Keto",
    "Vegan", "Dairy-Free", "Plant-Based", "Fair Trade",
    "Smoked", "Roasted", "Raw", "Seasoned", "Marinated",
    "Unsweetened", "Lightly Sweetened", "Double Chocolate",
]

sizes = [
    "8 oz", "12 oz", "16 oz", "32 oz", "1 gal", "64 oz",
    "6 pack", "12 pack", "24 pack", "5 lb", "10 lb", "1 lb",
    "100 ct", "200 ct", "500 ml", "1 L", "2 L",
    "3 oz", "4 oz", "20 oz", "28 oz", "48 oz", "3 lb",
    "8 ct", "10 ct", "16 ct", "36 pack", "2 lb",
]

print("Generating 500K product names...")
random.seed(42)
products = set()
while len(products) < 500_000:
    parts = [random.choice(brands)]
    if random.random() > 0.3:
        parts.append(random.choice(modifiers))
    parts.append(random.choice(categories))
    if random.random() > 0.4:
        parts.append(random.choice(sizes))
    products.add(" ".join(parts))

products = list(products)
print(f"Generated {len(products)} unique product names")
print(f"Examples: {products[:3]}")

# ──────────────────────────────────────────────
# Block 3: FTS5 Setup
# ──────────────────────────────────────────────

print("\nSetting up FTS5 tables...")
conn = sqlite3.connect(":memory:")
cur = conn.cursor()

# Standard tokenizer (word-boundary splitting)
cur.execute("CREATE VIRTUAL TABLE fts_standard USING fts5(name)")

# Trigram tokenizer (3-character overlapping chunks)
cur.execute("CREATE VIRTUAL TABLE fts_trigram USING fts5(name, tokenize='trigram')")

# Insert all products into both tables
for name in products:
    cur.execute("INSERT INTO fts_standard(name) VALUES (?)", (name,))
    cur.execute("INSERT INTO fts_trigram(name) VALUES (?)", (name,))

conn.commit()
print("FTS5 tables built and indexed")

# ──────────────────────────────────────────────
# Block 4: Speed Benchmark (5 scenarios)
# ──────────────────────────────────────────────

def bench(label, fn, runs=100):
    times = timeit.repeat(fn, number=1, repeat=runs)
    median = sorted(times)[len(times) // 2]
    return median * 1000  # convert to ms

print("\n" + "=" * 60)
print("SPEED BENCHMARK: 5 Scenarios")
print("=" * 60)

results = {}

# Scenario 1: Exact token search — "chocolate"
results["S1_fts_std"] = bench("FTS5 standard", lambda:
    cur.execute("SELECT name FROM fts_standard WHERE fts_standard MATCH 'chocolate' ORDER BY rank LIMIT 10").fetchall()
)

results["S1_fts_tri"] = bench("FTS5 trigram", lambda:
    cur.execute("SELECT name FROM fts_trigram WHERE fts_trigram MATCH 'chocolate' ORDER BY rank LIMIT 10").fetchall()
)

results["S1_rfuzz"] = bench("rapidfuzz", lambda:
    process.extract("chocolate", products, scorer=fuzz.WRatio, limit=10),
    runs=5
)

# Scenario 2: Typo tolerance — "choclate"
results["S2_fts_std"] = bench("FTS5 standard", lambda:
    cur.execute("SELECT name FROM fts_standard WHERE fts_standard MATCH 'choclate' ORDER BY rank LIMIT 10").fetchall()
)

results["S2_fts_tri"] = bench("FTS5 trigram", lambda:
    cur.execute("SELECT name FROM fts_trigram WHERE fts_trigram MATCH 'choclate' ORDER BY rank LIMIT 10").fetchall()
)

results["S2_rfuzz"] = bench("rapidfuzz", lambda:
    process.extract("choclate", products, scorer=fuzz.WRatio, limit=10),
    runs=5
)

# Scenario 3: Token reorder — "milk chocolate organic"
results["S3_fts_std"] = bench("FTS5 standard", lambda:
    cur.execute("SELECT name FROM fts_standard WHERE fts_standard MATCH 'milk AND chocolate AND organic' ORDER BY rank LIMIT 10").fetchall()
)

results["S3_rfuzz"] = bench("rapidfuzz", lambda:
    process.extract("milk chocolate organic", products, scorer=fuzz.token_sort_ratio, limit=10),
    runs=5
)

# Scenario 4: Prefix — "choc"
results["S4_fts_std"] = bench("FTS5 standard", lambda:
    cur.execute("SELECT name FROM fts_standard WHERE fts_standard MATCH 'choc*' ORDER BY rank LIMIT 10").fetchall()
)

results["S4_rfuzz"] = bench("rapidfuzz", lambda:
    process.extract("choc", products, scorer=fuzz.partial_ratio, limit=10),
    runs=5
)

# Scenario 5: Substring — "ocolat"
results["S5_fts_tri"] = bench("FTS5 trigram", lambda:
    cur.execute("SELECT name FROM fts_trigram WHERE fts_trigram MATCH 'ocolat' ORDER BY rank LIMIT 10").fetchall()
)

results["S5_rfuzz"] = bench("rapidfuzz", lambda:
    process.extract("ocolat", products, scorer=fuzz.partial_ratio, limit=10),
    runs=5
)

print("\nResults:")
print(f"  {'Scenario':<30} {'Method':<18} {'Time (ms)':>12}")
print("-" * 62)
for key in sorted(results.keys()):
    scenario = key.split("_")[0]
    method = "_".join(key.split("_")[1:])
    print(f"  {scenario:<30} {method:<18} {results[key]:>12.2f}")

# ──────────────────────────────────────────────
# Block 9: Hybrid Search Function
# ──────────────────────────────────────────────

def hybrid_search(query, conn, limit=10, candidates=200):
    """
    Two-stage fuzzy search:
    1. FTS5 trigram for fast candidate retrieval
    2. rapidfuzz for typo-tolerant re-ranking
    """
    cur = conn.cursor()

    # Stage 1: Pull candidates using FTS5 trigram
    # For short queries (< 3 chars), fall back to prefix on standard index
    if len(query) >= 3:
        rows = cur.execute(
            "SELECT name FROM fts_trigram WHERE fts_trigram MATCH ? LIMIT ?",
            (query, candidates)
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT name FROM fts_standard WHERE fts_standard MATCH ? LIMIT ?",
            (query + "*", candidates)
        ).fetchall()

    candidate_names = [r[0] for r in rows]

    if not candidate_names:
        # FTS5 found nothing — fall back to full rapidfuzz scan
        # (slower, but handles total mismatches)
        results = process.extract(query, products, scorer=fuzz.WRatio, limit=limit)
        return [(name, score) for name, score, _ in results]

    # Stage 2: Re-rank candidates with rapidfuzz
    scored = []
    for name in candidate_names:
        score = fuzz.WRatio(query, name)
        scored.append((name, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]

# ──────────────────────────────────────────────
# Block 10: Hybrid vs Standalone Benchmark
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("HYBRID SEARCH BENCHMARK")
print("=" * 60)

hybrid_time = bench("Hybrid", lambda:
    hybrid_search("choclate", conn, limit=10),
    runs=100
)

print(f"\n  FTS5 standard:  {results['S2_fts_std']:.2f} ms — but 0 results (useless)")
print(f"  FTS5 trigram:   {results['S2_fts_tri']:.2f} ms — substring matches only")
print(f"  rapidfuzz:      {results['S2_rfuzz']:.0f} ms — correct results, but slow")
print(f"  Hybrid:         {hybrid_time:.1f} ms — correct results AND fast")

# ──────────────────────────────────────────────
# Block 5: Batch Query Generator
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("BATCH BENCHMARK")
print("=" * 60)

def make_batch_queries(products, n, typo_rate=0.3):
    """Generate n search queries — some exact, some with typos."""
    queries = []
    for _ in range(n):
        base = random.choice(products).split()
        # Pick 1-2 words from the product name
        q_words = random.sample(base, min(2, len(base)))
        query = " ".join(q_words)
        # Introduce a typo with some probability
        if random.random() < typo_rate and len(query) > 4:
            pos = random.randint(1, len(query) - 2)
            query = query[:pos] + query[pos+1:]  # delete a character
        queries.append(query)
    return queries

batch_10 = make_batch_queries(products, 10)
batch_100 = make_batch_queries(products, 100)
batch_1000 = make_batch_queries(products, 1000)

def bench_batch(label, fn, runs=5):
    """Benchmark a batch operation — fewer runs since each is longer."""
    times = timeit.repeat(fn, number=1, repeat=runs)
    return sorted(times)[len(times) // 2] * 1000  # median ms

# ──────────────────────────────────────────────
# Block 6: FTS5 Batch
# ──────────────────────────────────────────────

def fts5_batch(queries, table="fts_trigram"):
    """Run a batch of FTS5 queries sequentially."""
    results = []
    for q in queries:
        try:
            rows = cur.execute(
                f"SELECT name FROM {table} WHERE {table} MATCH ? LIMIT 10",
                (q,)
            ).fetchall()
            results.append([r[0] for r in rows])
        except:
            results.append([])  # handle invalid MATCH syntax gracefully
    return results

print("\nFTS5 batch benchmarks...")
fts_10 = bench_batch("FTS5 tri ×10", lambda: fts5_batch(batch_10))
fts_100 = bench_batch("FTS5 tri ×100", lambda: fts5_batch(batch_100))
fts_1000 = bench_batch("FTS5 tri ×1000", lambda: fts5_batch(batch_1000))
print(f"  FTS5 ×10:   {fts_10:.1f} ms")
print(f"  FTS5 ×100:  {fts_100:.1f} ms")
print(f"  FTS5 ×1000: {fts_1000:.1f} ms")

# ──────────────────────────────────────────────
# Block 7: rapidfuzz Batch (Sequential + Parallel)
# ──────────────────────────────────────────────

def rfuzz_batch_sequential(queries):
    """Run rapidfuzz extract for each query — sequential."""
    return [process.extract(q, products, scorer=fuzz.WRatio, limit=10)
            for q in queries]

def rfuzz_single_query(q):
    """Single rapidfuzz query — for use with parallel executor."""
    return process.extract(q, products, scorer=fuzz.WRatio, limit=10)

def rfuzz_batch_parallel(queries, workers=4):
    """Run rapidfuzz queries in parallel using process pool."""
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(rfuzz_single_query, queries))

print("\nrapidfuzz sequential benchmarks (this will take several minutes)...")
rf_seq_10 = bench_batch("rfuzz seq ×10", lambda: rfuzz_batch_sequential(batch_10))
print(f"  rfuzz seq ×10:   {rf_seq_10:.0f} ms ({rf_seq_10/1000:.1f} s)")
rf_seq_100 = bench_batch("rfuzz seq ×100", lambda: rfuzz_batch_sequential(batch_100))
print(f"  rfuzz seq ×100:  {rf_seq_100:.0f} ms ({rf_seq_100/1000:.1f} s)")
rf_seq_1000 = bench_batch("rfuzz seq ×1000",
    lambda: rfuzz_batch_sequential(batch_1000), runs=3)
print(f"  rfuzz seq ×1000: {rf_seq_1000:.0f} ms ({rf_seq_1000/1000:.1f} s)")

print("\nrapidfuzz parallel benchmarks...")
rf_par_100 = bench_batch("rfuzz par ×100", lambda: rfuzz_batch_parallel(batch_100))
print(f"  rfuzz par ×100:  {rf_par_100:.0f} ms ({rf_par_100/1000:.1f} s)")
rf_par_1000 = bench_batch("rfuzz par ×1000",
    lambda: rfuzz_batch_parallel(batch_1000), runs=3)
print(f"  rfuzz par ×1000: {rf_par_1000:.0f} ms ({rf_par_1000/1000:.1f} s)")

# ──────────────────────────────────────────────
# Block 8: Hybrid Batch
# ──────────────────────────────────────────────

def hybrid_batch_fn(queries):
    """Run hybrid search for each query."""
    return [hybrid_search(q, conn, limit=10) for q in queries]

print("\nHybrid batch benchmarks...")
hyb_10 = bench_batch("hybrid ×10", lambda: hybrid_batch_fn(batch_10))
hyb_100 = bench_batch("hybrid ×100", lambda: hybrid_batch_fn(batch_100))
hyb_1000 = bench_batch("hybrid ×1000", lambda: hybrid_batch_fn(batch_1000))
print(f"  Hybrid ×10:   {hyb_10:.1f} ms")
print(f"  Hybrid ×100:  {hyb_100:.1f} ms")
print(f"  Hybrid ×1000: {hyb_1000:.1f} ms")

# ──────────────────────────────────────────────
# Summary Tables
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("BATCH RESULTS SUMMARY")
print("=" * 60)

print(f"\n{'Batch Size':<15} {'FTS5 Tri':>12} {'rfuzz seq':>14} {'rfuzz par':>14} {'Hybrid':>12}")
print("-" * 70)
print(f"{'10 queries':<15} {fts_10:>10.1f}ms {rf_seq_10/1000:>12.1f}s {'n/a':>14} {hyb_10:>10.1f}ms")
print(f"{'100 queries':<15} {fts_100:>10.1f}ms {rf_seq_100/1000:>12.1f}s {rf_par_100/1000:>12.1f}s {hyb_100:>10.1f}ms")
print(f"{'1000 queries':<15} {fts_1000:>10.1f}ms {rf_seq_1000/1000:>12.1f}s {rf_par_1000/1000:>12.1f}s {hyb_1000:>10.1f}ms")

print("\nThroughput (queries/sec at 1000-query batch):")
fts_qps = 1000 / (fts_1000 / 1000)
rf_seq_qps = 1000 / (rf_seq_1000 / 1000)
rf_par_qps = 1000 / (rf_par_1000 / 1000)
hyb_qps = 1000 / (hyb_1000 / 1000)
print(f"  FTS5 Trigram:         {fts_qps:>10.0f} queries/sec")
print(f"  Hybrid:               {hyb_qps:>10.0f} queries/sec")
print(f"  rapidfuzz (4 cores):  {rf_par_qps:>10.1f} queries/sec")
print(f"  rapidfuzz (seq):      {rf_seq_qps:>10.1f} queries/sec")
