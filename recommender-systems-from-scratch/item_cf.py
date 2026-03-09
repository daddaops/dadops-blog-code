"""Item-Based Collaborative Filtering."""
from helpers import ratings, n_users, n_items, user_avgs, names

def item_cosine_sim(i1, i2, ratings):
    """Cosine similarity between two items based on users who rated both."""
    co_users = [u for u in range(len(ratings))
                if ratings[u][i1] > 0 and ratings[u][i2] > 0]
    if len(co_users) < 2:
        return 0.0

    mean1 = sum(ratings[u][i1] for u in co_users) / len(co_users)
    mean2 = sum(ratings[u][i2] for u in co_users) / len(co_users)

    num = sum((ratings[u][i1] - mean1) * (ratings[u][i2] - mean2)
              for u in co_users)
    d1 = sum((ratings[u][i1] - mean1)**2 for u in co_users) ** 0.5
    d2 = sum((ratings[u][i2] - mean2)**2 for u in co_users) ** 0.5
    return num / (d1 * d2) if d1 * d2 > 0 else 0.0

# Precompute item-item similarity matrix
item_sims = [[0.0] * n_items for _ in range(n_items)]
for i in range(n_items):
    for j in range(i + 1, n_items):
        s = item_cosine_sim(i, j, ratings)
        item_sims[i][j] = s
        item_sims[j][i] = s

def predict_item_cf(user, item, ratings, k=3):
    """Predict rating from user's ratings of the k most similar items."""
    if ratings[user][item] > 0:
        return ratings[user][item]

    rated_items = [j for j in range(n_items)
                   if j != item and ratings[user][j] > 0]
    if not rated_items:
        return user_avgs[user]

    sims = [(j, item_sims[item][j]) for j in rated_items]
    sims.sort(key=lambda x: -x[1])
    top_k = [(j, s) for j, s in sims[:k] if s > 0]

    if not top_k:
        return user_avgs[user]

    num = sum(s * ratings[user][j] for j, s in top_k)
    den = sum(abs(s) for _, s in top_k)
    return num / den

# Evaluate item-based CF
errors_icf = []
for i in range(n_users):
    for j in range(n_items):
        if ratings[i][j] > 0:
            true_val = ratings[i][j]
            ratings[i][j] = 0
            pred = predict_item_cf(i, j, ratings, k=3)
            ratings[i][j] = true_val
            errors_icf.append((true_val - pred) ** 2)

rmse_icf = (sum(errors_icf) / len(errors_icf)) ** 0.5
print(f"Item-CF RMSE: {rmse_icf:.3f}")
print(f"\nItem similarities (Nebula Quest vs ...):")
for j in range(1, n_items):
    print(f"  {names[j]:<14s} {item_sims[0][j]:+.3f}")
