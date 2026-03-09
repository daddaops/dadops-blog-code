"""The Rating Matrix: baselines for recommendation."""
from helpers import ratings, n_users, n_items, global_avg, user_avgs

# Count rated entries
rated = sum(1 for u in ratings for r in u if r > 0)
total = n_users * n_items
print(f"Rated: {rated}/{total} ({100*rated/total:.1f}%)")
print(f"Sparsity: {100*(1 - rated/total):.1f}%")

print(f"\nGlobal average rating: {global_avg:.2f}")

# RMSE of naive baseline on known ratings
mse = sum((r - global_avg)**2 for u in ratings for r in u if r > 0)
rmse_naive = (mse / rated) ** 0.5
print(f"Naive baseline RMSE: {rmse_naive:.3f}")

mse_user = sum((r - user_avgs[i])**2
               for i, u in enumerate(ratings) for r in u if r > 0)
rmse_user = (mse_user / rated) ** 0.5
print(f"Per-user avg RMSE:   {rmse_user:.3f}")
