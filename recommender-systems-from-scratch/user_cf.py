"""User-Based Collaborative Filtering."""
from helpers import ratings, n_users, n_items, user_avgs

def cosine_sim(u1, u2, ratings):
    """Cosine similarity between two users on co-rated items, mean-centered."""
    r1, r2 = ratings[u1], ratings[u2]
    co_rated = [j for j in range(len(r1)) if r1[j] > 0 and r2[j] > 0]
    if len(co_rated) < 2:
        return 0.0

    mean1 = sum(r1[j] for j in co_rated) / len(co_rated)
    mean2 = sum(r2[j] for j in co_rated) / len(co_rated)

    num = sum((r1[j] - mean1) * (r2[j] - mean2) for j in co_rated)
    d1 = sum((r1[j] - mean1)**2 for j in co_rated) ** 0.5
    d2 = sum((r2[j] - mean2)**2 for j in co_rated) ** 0.5
    return num / (d1 * d2) if d1 * d2 > 0 else 0.0

def predict_user_cf(user, item, ratings, k=3):
    """Predict rating using k most similar users who rated this item."""
    if ratings[user][item] > 0:
        return ratings[user][item]

    candidates = [u for u in range(len(ratings))
                  if u != user and ratings[u][item] > 0]
    if not candidates:
        return user_avgs[user]

    sims = [(u, cosine_sim(user, u, ratings)) for u in candidates]
    sims.sort(key=lambda x: -x[1])
    top_k = [(u, s) for u, s in sims[:k] if s > 0]

    if not top_k:
        return user_avgs[user]

    num = sum(s * ratings[u][item] for u, s in top_k)
    den = sum(abs(s) for _, s in top_k)
    return num / den

# Predict User 0's rating for Movie 4 (a comedy)
pred = predict_user_cf(0, 4, ratings, k=3)
print(f"User 0 rating for 'Dad Joke: The Movie': {pred:.2f}")

# RMSE on all known ratings (leave-one-out style)
errors = []
for i in range(n_users):
    for j in range(n_items):
        if ratings[i][j] > 0:
            true_val = ratings[i][j]
            ratings[i][j] = 0
            pred = predict_user_cf(i, j, ratings, k=3)
            ratings[i][j] = true_val
            errors.append((true_val - pred) ** 2)

rmse_ucf = (sum(errors) / len(errors)) ** 0.5
print(f"User-CF RMSE: {rmse_ucf:.3f}")
