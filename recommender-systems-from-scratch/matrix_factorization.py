"""Matrix Factorization via SGD."""
import random
from helpers import ratings, n_users, n_items, global_avg, observed

random.seed(42)
k = 5          # latent factors
lr = 0.01      # learning rate
reg = 0.1      # regularization strength
epochs = 80

# Initialize randomly
U = [[random.gauss(0, 0.1) for _ in range(k)] for _ in range(n_users)]
V = [[random.gauss(0, 0.1) for _ in range(k)] for _ in range(n_items)]
bu = [0.0] * n_users   # user biases
bi = [0.0] * n_items   # item biases
mu = global_avg         # global mean

# Need a mutable copy of observed for shuffling
obs = list(observed)

for epoch in range(epochs):
    random.shuffle(obs)
    total_loss = 0.0
    for i, j, r in obs:
        # Predict: mu + bu[i] + bi[j] + dot(U[i], V[j])
        dot = sum(U[i][f] * V[j][f] for f in range(k))
        pred = mu + bu[i] + bi[j] + dot
        err = r - pred
        total_loss += err ** 2

        # Update biases
        bu[i] += lr * (err - reg * bu[i])
        bi[j] += lr * (err - reg * bi[j])

        # Update latent factors
        for f in range(k):
            u_old = U[i][f]
            U[i][f] += lr * (err * V[j][f] - reg * U[i][f])
            V[j][f] += lr * (err * u_old - reg * V[j][f])

    if (epoch + 1) % 20 == 0:
        rmse = (total_loss / len(obs)) ** 0.5
        print(f"Epoch {epoch+1:3d}: RMSE = {rmse:.4f}")

# Final RMSE
final_loss = sum((r - mu - bu[i] - bi[j] -
                  sum(U[i][f]*V[j][f] for f in range(k)))**2
                 for i, j, r in obs)
rmse_mf = (final_loss / len(obs)) ** 0.5
print(f"\nMatrix Factorization RMSE: {rmse_mf:.3f}")
print(f"\nUser biases: {[f'{b:+.2f}' for b in bu]}")
print(f"Item biases: {[f'{b:+.2f}' for b in bi]}")
