"""
Code Block 3: Embedding-based router with logistic regression.

From: https://dadops.dev/blog/llm-model-routing/

Trains a logistic regression classifier on sentence embeddings to learn
query-to-tier mappings from labeled data. Uses all-MiniLM-L6-v2 (~80MB).

Requires: sentence-transformers, scikit-learn
No API key required (but downloads model on first run).
"""

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load a small, fast embedding model (~80MB)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Your labeled dataset: (query, correct_tier)
#    In practice, build this by running samples through all tiers
#    and having a judge pick the cheapest tier with equivalent quality.
labeled_data = [
    ("What is my account balance?", 1),
    ("Extract the customer name from this email", 1),
    ("Summarize the key points from this meeting transcript", 2),
    ("Write a professional follow-up email to this lead", 2),
    ("Analyze why our churn rate spiked last quarter and "
     "propose three retention strategies with expected ROI", 3),
    # ... hundreds more in practice
]

queries, tiers = zip(*labeled_data)
embeddings = embedder.encode(list(queries))

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, tiers, test_size=0.2, random_state=42, stratify=tiers
)

# 3. Logistic regression on 384-dim embeddings — trains in milliseconds
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)


# 4. Route new queries
def embedding_route(query: str) -> int:
    vec = embedder.encode([query])
    return int(clf.predict(vec)[0])


if __name__ == "__main__":
    # 5. Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Tier1", "Tier2", "Tier3"]))

    # Test routing
    test_queries = [
        "What is the refund policy?",
        "Summarize last month's sales report",
        "Analyze the root cause of our latency spike and propose a fix",
    ]
    for q in test_queries:
        print(f"Tier {embedding_route(q)}: {q[:60]}")
