from collections import defaultdict

# A knowledge graph: list of (head, relation, tail) triples
triples = [
    ("Einstein", "born_in", "Ulm"),
    ("Einstein", "field", "Physics"),
    ("Einstein", "won", "Nobel_Physics"),
    ("Curie", "born_in", "Warsaw"),
    ("Curie", "field", "Physics"),
    ("Curie", "field", "Chemistry"),
    ("Curie", "won", "Nobel_Physics"),
    ("Curie", "won", "Nobel_Chemistry"),
    ("Bohr", "born_in", "Copenhagen"),
    ("Bohr", "field", "Physics"),
    ("Bohr", "won", "Nobel_Physics"),
    ("Ulm", "located_in", "Germany"),
    ("Warsaw", "located_in", "Poland"),
    ("Copenhagen", "located_in", "Denmark"),
]

if __name__ == "__main__":
    # Build adjacency: entity -> [(relation, neighbor)]
    graph = defaultdict(list)
    for h, r, t in triples:
        graph[h].append((r, t))

    # One-hop: "What did Einstein win?"
    wins = [t for r, t in graph["Einstein"] if r == "won"]
    print(f"Einstein won: {wins}")
    # -> Einstein won: ['Nobel_Physics']

    # Two-hop: "What country was Einstein born in?"
    cities = [t for r, t in graph["Einstein"] if r == "born_in"]
    countries = [t for c in cities for r, t in graph[c] if r == "located_in"]
    print(f"Einstein's birth country: {countries}")
    # -> Einstein's birth country: ['Germany']
