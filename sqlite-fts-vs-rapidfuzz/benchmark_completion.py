"""
Completion script: runs ONLY the parts that were missing from the
previous interrupted benchmark_all.py run.

Missing:
  - rfuzz parallel ×1000
  - Hybrid batch (×10, ×100, ×1000)
  - Summary table with all results combined
"""
import random
import sqlite3
import os
import time
import timeit
from rapidfuzz import fuzz, process
from concurrent.futures import ProcessPoolExecutor

def log(msg=""):
    print(msg, flush=True)

# ── Reproduce the same dataset (same seed) ──────────────
log("Generating 500K product names (seed=42)...")
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
log(f"Generated {len(products)} products")

# ── FTS5 setup ──────────────────────────────────
log("Setting up FTS5 tables...")
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE VIRTUAL TABLE fts_standard USING fts5(name)")
cur.execute("CREATE VIRTUAL TABLE fts_trigram USING fts5(name, tokenize='trigram')")
for name in products:
    cur.execute("INSERT INTO fts_standard(name) VALUES (?)", (name,))
for name in products:
    cur.execute("INSERT INTO fts_trigram(name) VALUES (?)", (name,))
conn.commit()
log("FTS5 tables ready")

# ── Helpers ──────────────────────────────────────
def bench_batch(label, fn, runs=5):
    times = timeit.repeat(fn, number=1, repeat=runs)
    return sorted(times)[len(times) // 2] * 1000

def make_batch_queries(products, n, typo_rate=0.3):
    queries = []
    for _ in range(n):
        base = random.choice(products).split()
        q_words = random.sample(base, min(2, len(base)))
        query = " ".join(q_words)
        if random.random() < typo_rate and len(query) > 4:
            pos = random.randint(1, len(query) - 2)
            query = query[:pos] + query[pos+1:]
        queries.append(query)
    return queries

def _fts5_safe(query):
    """Escape FTS5 special characters so MATCH doesn't crash."""
    # Remove characters that FTS5 interprets as syntax
    import re
    return re.sub(r'[%*"^$(){}:+\-]', ' ', query).strip()

def hybrid_search(query, conn, limit=10, candidates=200):
    cur = conn.cursor()
    safe_q = _fts5_safe(query)
    if not safe_q or len(safe_q) < 2:
        # Too short after escaping — full scan
        results = process.extract(query, products, scorer=fuzz.WRatio, limit=limit)
        return [(name, score) for name, score, _ in results]
    try:
        if len(safe_q) >= 3:
            rows = cur.execute(
                "SELECT name FROM fts_trigram WHERE fts_trigram MATCH ? LIMIT ?",
                (safe_q, candidates)
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT name FROM fts_standard WHERE fts_standard MATCH ? LIMIT ?",
                (safe_q + "*", candidates)
            ).fetchall()
    except sqlite3.OperationalError:
        # Fallback for any remaining FTS5 syntax issues
        results = process.extract(query, products, scorer=fuzz.WRatio, limit=limit)
        return [(name, score) for name, score, _ in results]
    candidate_names = [r[0] for r in rows]
    if not candidate_names:
        results = process.extract(query, products, scorer=fuzz.WRatio, limit=limit)
        return [(name, score) for name, score, _ in results]
    scored = []
    for name in candidate_names:
        score = fuzz.WRatio(query, name)
        scored.append((name, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]

# ── Generate batches (same seed state continues) ──
batch_10 = make_batch_queries(products, 10)
batch_100 = make_batch_queries(products, 100)
batch_1000 = make_batch_queries(products, 1000)

# ── MISSING PIECE 1: Hybrid batch ──────────────
log("\n" + "=" * 60)
log("HYBRID BATCH BENCHMARK")
log("=" * 60)

def hybrid_batch_fn(queries):
    return [hybrid_search(q, conn, limit=10) for q in queries]

log("Running hybrid batch ×10...")
hyb_10 = bench_batch("hybrid ×10", lambda: hybrid_batch_fn(batch_10), runs=2)
log(f"  Hybrid ×10:   {hyb_10:.1f} ms")

log("Running hybrid batch ×100...")
hyb_100 = bench_batch("hybrid ×100", lambda: hybrid_batch_fn(batch_100), runs=2)
log(f"  Hybrid ×100:  {hyb_100:.1f} ms")

log("Running hybrid batch ×1000 (single run)...")
t0 = time.perf_counter()
hybrid_batch_fn(batch_1000)
hyb_1000 = (time.perf_counter() - t0) * 1000
log(f"  Hybrid ×1000: {hyb_1000:.1f} ms")

# ── MISSING PIECE 2: rfuzz parallel ×1000 ──────
log("\n" + "=" * 60)
log("RAPIDFUZZ PARALLEL x1000")
log("=" * 60)

def rfuzz_single_query(q):
    return process.extract(q, products, scorer=fuzz.WRatio, limit=10)

n_workers = os.cpu_count() or 4
log(f"Using {n_workers} workers")
log("Running rfuzz parallel ×1000 (single measurement)...")
t0 = time.perf_counter()
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    list(executor.map(rfuzz_single_query, batch_1000))
rf_par_1000 = (time.perf_counter() - t0) * 1000
log(f"  rfuzz par ×1000: {rf_par_1000:.0f} ms ({rf_par_1000/1000:.1f} s)")

# ── SUMMARY TABLE (combining previous + new results) ──
log("\n" + "=" * 60)
log("COMBINED SUMMARY (previous run + completion)")
log("=" * 60)

# Values from the previous partial run:
prev = {
    "fts_10": 3.1, "fts_100": 39.7, "fts_1000": 357.0,
    "rf_seq_10": 26665, "rf_seq_100": 282816,
    "rf_par_100": 56449,
}
prev["rf_seq_1000"] = prev["rf_seq_100"] * 10  # extrapolated

log(f"\n{'Batch Size':<15} {'FTS5 Tri':>12} {'rfuzz seq':>14} {'rfuzz par':>14} {'Hybrid':>12}")
log("-" * 70)
log(f"{'10 queries':<15} {prev['fts_10']:>10.1f}ms {prev['rf_seq_10']/1000:>12.1f}s {'n/a':>14} {hyb_10:>10.1f}ms")
log(f"{'100 queries':<15} {prev['fts_100']:>10.1f}ms {prev['rf_seq_100']/1000:>12.1f}s {prev['rf_par_100']/1000:>12.1f}s {hyb_100:>10.1f}ms")
log(f"{'1000 queries':<15} {prev['fts_1000']:>10.1f}ms {prev['rf_seq_1000']/1000:>12.1f}s {rf_par_1000/1000:>12.1f}s {hyb_1000:>10.1f}ms")

log("\nThroughput (queries/sec at 1000-query batch):")
fts_qps = 1000 / (prev["fts_1000"] / 1000)
rf_seq_qps = 1000 / (prev["rf_seq_1000"] / 1000)
rf_par_qps = 1000 / (rf_par_1000 / 1000)
hyb_qps = 1000 / (hyb_1000 / 1000)
log(f"  FTS5 Trigram:         {fts_qps:>10.0f} queries/sec")
log(f"  Hybrid:               {hyb_qps:>10.0f} queries/sec")
log(f"  rapidfuzz ({n_workers} cores):  {rf_par_qps:>10.1f} queries/sec")
log(f"  rapidfuzz (seq):      {rf_seq_qps:>10.1f} queries/sec")

# ── Check how many hybrid queries fall back to full scan ──
log("\n" + "=" * 60)
log("HYBRID FALLBACK ANALYSIS")
log("=" * 60)
fallback_count = 0
for q in batch_1000:
    if len(q) >= 3:
        rows = cur.execute(
            "SELECT COUNT(*) FROM fts_trigram WHERE fts_trigram MATCH ? LIMIT 200",
            (q,)
        ).fetchone()[0]
    else:
        rows = cur.execute(
            "SELECT COUNT(*) FROM fts_standard WHERE fts_standard MATCH ? LIMIT 200",
            (q + "*",)
        ).fetchone()[0]
    if rows == 0:
        fallback_count += 1
log(f"  Queries with 0 FTS5 candidates (full fallback): {fallback_count}/{len(batch_1000)}")
log(f"  Percentage: {fallback_count/len(batch_1000)*100:.1f}%")

log("\nDone!")
