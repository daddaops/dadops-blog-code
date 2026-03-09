import re
from collections import defaultdict

def extract_triples(sentences):
    """Extract (subject, predicate, object) triples from text."""
    patterns = [
        (r"(\w+) was born in (\w+)",
         lambda m: (m.group(1), "born_in", m.group(2))),
        (r"(\w+) discovered (\w+)",
         lambda m: (m.group(1), "discovered", m.group(2))),
        (r"(\w+) works at (\w+)",
         lambda m: (m.group(1), "works_at", m.group(2))),
        (r"(\w+) is a (\w+)",
         lambda m: (m.group(1), "is_a", m.group(2))),
    ]
    extracted = []
    for sent in sentences:
        for pattern, extractor in patterns:
            match = re.search(pattern, sent)
            if match:
                extracted.append(extractor(match))
    return extracted

sentences = [
    "Einstein was born in Ulm",
    "Einstein discovered Relativity",
    "Relativity is a Theory",
    "Curie discovered Radium",
    "Curie works at Sorbonne",
]

kg = extract_triples(sentences)
for h, r, t in kg:
    print(f"  ({h}, {r}, {t})")

# Two-hop: "What type of thing did Einstein discover?"
graph = defaultdict(list)
for h, r, t in kg:
    graph[h].append((r, t))

mid = [t for r, t in graph["Einstein"] if r == "discovered"]
answer = [t for m in mid for r, t in graph[m] if r == "is_a"]
print(f"\nEinstein discovered a {answer[0]}.")
# -> Einstein discovered a Theory.
