"""Content-Based Filtering with TF-IDF."""
import math
from helpers import ratings, n_items, names

# Item features: [action, comedy, drama, sci-fi, romance, thriller]
features = [
    [0.1, 0.0, 0.0, 0.9, 0.0, 0.2],  # Nebula Quest (sci-fi)
    [0.2, 0.0, 0.0, 0.8, 0.0, 0.3],  # Star Forge (sci-fi)
    [0.3, 0.0, 0.0, 0.7, 0.0, 0.4],  # Void Runners (sci-fi/action)
    [0.0, 0.9, 0.1, 0.0, 0.1, 0.0],  # Laugh Track (comedy)
    [0.0, 0.8, 0.2, 0.0, 0.2, 0.0],  # Dad Joke: The Movie (comedy)
    [0.0, 0.9, 0.0, 0.0, 0.0, 0.1],  # Pun Intended (comedy)
    [0.0, 0.1, 0.9, 0.0, 0.3, 0.0],  # Quiet River (drama)
    [0.0, 0.0, 0.9, 0.0, 0.4, 0.0],  # The Long Goodbye (drama)
    [0.9, 0.0, 0.1, 0.2, 0.0, 0.7],  # Fist Planet (action)
    [0.8, 0.1, 0.0, 0.1, 0.0, 0.8],  # Thunder Road (action)
]
n_features = len(features[0])

# TF-IDF weighting
doc_freq = [sum(1 for item in features if item[f] > 0.3)
            for f in range(n_features)]
n_docs = len(features)
idf = [math.log(n_docs / (1 + df)) for df in doc_freq]

# Apply IDF weights
tfidf = [[features[i][f] * idf[f] for f in range(n_features)]
         for i in range(n_items)]

def build_user_profile(user, ratings, tfidf):
    """Weighted average of liked items' TF-IDF vectors."""
    profile = [0.0] * n_features
    weight_sum = 0.0
    for j in range(n_items):
        if ratings[user][j] >= 3:
            w = ratings[user][j]
            for f in range(n_features):
                profile[f] += w * tfidf[j][f]
            weight_sum += w
    if weight_sum > 0:
        profile = [p / weight_sum for p in profile]
    return profile

def content_score(profile, item_vec):
    """Cosine similarity between user profile and item."""
    dot = sum(a * b for a, b in zip(profile, item_vec))
    na = sum(a**2 for a in profile) ** 0.5
    nb = sum(b**2 for b in item_vec) ** 0.5
    return dot / (na * nb) if na * nb > 0 else 0.0

# Score all unrated items for User 0
profile_u0 = build_user_profile(0, ratings, tfidf)
print("User 0's profile (sci-fi/action fan):")
genre_names = ["action", "comedy", "drama", "sci-fi", "romance", "thriller"]
for f in range(n_features):
    if profile_u0[f] > 0.01:
        print(f"  {genre_names[f]:<10s} {profile_u0[f]:.3f}")

print("\nContent-based scores for User 0's unrated movies:")
for j in range(n_items):
    if ratings[0][j] == 0:
        score = content_score(profile_u0, tfidf[j])
        print(f"  {names[j]:<14s} score={score:.3f}")
