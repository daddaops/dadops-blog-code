"""Shared data and helpers for recommender system scripts."""

# 8 users x 10 movies, ratings 1-5 (0 = unrated)
ratings = [
    [5, 4, 5, 1, 0, 0, 2, 0, 4, 3],  # User 0: sci-fi + action fan
    [4, 5, 4, 2, 1, 2, 0, 1, 3, 4],  # User 1: sci-fi + action fan
    [1, 2, 1, 5, 4, 5, 3, 0, 0, 1],  # User 2: comedy lover
    [2, 1, 0, 4, 5, 4, 0, 3, 1, 0],  # User 3: comedy lover
    [0, 1, 0, 2, 0, 0, 5, 4, 0, 2],  # User 4: drama fan
    [2, 0, 1, 0, 3, 0, 4, 5, 0, 1],  # User 5: drama + some comedy
    [5, 4, 4, 1, 1, 0, 2, 0, 5, 5],  # User 6: sci-fi + action
    [0, 1, 2, 5, 4, 5, 2, 0, 0, 1],  # User 7: comedy lover
]
n_users = len(ratings)
n_items = len(ratings[0])

names = ["Nebula Quest", "Star Forge", "Void Runners",
         "Laugh Track", "Dad Joke", "Pun Intended",
         "Quiet River", "Long Goodbye", "Fist Planet", "Thunder Road"]

all_ratings = [r for u in ratings for r in u if r > 0]
global_avg = sum(all_ratings) / len(all_ratings)

user_avgs = []
for u in ratings:
    vals = [r for r in u if r > 0]
    user_avgs.append(sum(vals) / len(vals) if vals else global_avg)

observed = [(i, j, ratings[i][j])
            for i in range(n_users) for j in range(n_items)
            if ratings[i][j] > 0]
