"""
rapidfuzz basics — pairwise similarity and search demo.

Blog post: https://dadops.dev/blog/sqlite-fts-vs-rapidfuzz/
Code Block 1 from "SQLite FTS5 vs rapidfuzz: Fuzzy Search Showdown"
"""
from rapidfuzz import fuzz, process

if __name__ == "__main__":
    # Pairwise similarity
    print("Pairwise similarity:")
    r1 = fuzz.ratio("chocolate milk", "choclate milk")
    print(f"  fuzz.ratio('chocolate milk', 'choclate milk') = {r1:.1f}")

    r2 = fuzz.token_sort_ratio("milk chocolate", "chocolate milk")
    print(f"  fuzz.token_sort_ratio('milk chocolate', 'chocolate milk') = {r2:.1f}")

    # Search a list for the best match
    print("\nSearch a list:")
    products = ["Chocolate Milk 2%", "Dark Chocolate Bar 70%", "Whole Milk Organic"]
    result = process.extractOne("choclate milk", products)
    print(f"  process.extractOne('choclate milk', products) = {result}")
