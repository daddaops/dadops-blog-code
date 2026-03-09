"""Neural Collaborative Filtering (pure Python)."""
import random
from helpers import n_users, n_items, observed

random.seed(42)
emb_dim = 5     # embedding dimension
h1_size = 16    # first hidden layer
h2_size = 8     # second hidden layer

# Initialize weights (Xavier-like)
def rand_matrix(rows, cols, scale=None):
    if scale is None:
        scale = (2.0 / (rows + cols)) ** 0.5
    return [[random.gauss(0, scale) for _ in range(cols)]
            for _ in range(rows)]

# Embeddings
E_user = rand_matrix(n_users, emb_dim)
E_item = rand_matrix(n_items, emb_dim)
# Hidden layers: input = 2*emb_dim, output = 1
W1 = rand_matrix(2 * emb_dim, h1_size)
b1 = [0.0] * h1_size
W2 = rand_matrix(h1_size, h2_size)
b2 = [0.0] * h2_size
W3 = rand_matrix(h2_size, 1)
b3 = [0.0]

def relu(x): return max(0.0, x)

def forward(user, item):
    """Forward pass: embed, concat, 2 hidden layers, linear output."""
    x = E_user[user] + E_item[item]  # concatenated input
    # Hidden layer 1
    h1 = [relu(sum(x[i] * W1[i][j] for i in range(2*emb_dim)) + b1[j])
          for j in range(h1_size)]
    # Hidden layer 2
    h2 = [relu(sum(h1[i] * W2[i][j] for i in range(h1_size)) + b2[j])
          for j in range(h2_size)]
    # Output
    out = sum(h2[i] * W3[i][0] for i in range(h2_size)) + b3[0]
    return out, x, h1, h2

# Train with SGD and backpropagation
obs = list(observed)
lr_ncf = 0.005
for epoch in range(100):
    random.shuffle(obs)
    total_loss = 0.0
    for u, j, r in obs:
        pred, x, h1, h2 = forward(u, j)
        err = r - pred
        total_loss += err ** 2

        # Backprop through output layer
        d_out = -2 * err
        for i in range(h2_size):
            grad_w3 = d_out * h2[i]
            W3[i][0] -= lr_ncf * grad_w3
        b3[0] -= lr_ncf * d_out

        # Backprop through hidden layer 2
        d_h2 = [d_out * W3[i][0] * (1 if h2[i] > 0 else 0)
                for i in range(h2_size)]
        for i in range(h1_size):
            for o in range(h2_size):
                W2[i][o] -= lr_ncf * d_h2[o] * h1[i]
        for o in range(h2_size):
            b2[o] -= lr_ncf * d_h2[o]

        # Backprop through hidden layer 1
        d_h1 = [sum(d_h2[o] * W2[i][o] for o in range(h2_size))
                * (1 if h1[i] > 0 else 0) for i in range(h1_size)]
        for i in range(2 * emb_dim):
            for o in range(h1_size):
                W1[i][o] -= lr_ncf * d_h1[o] * x[i]
        for o in range(h1_size):
            b1[o] -= lr_ncf * d_h1[o]

        # Update embeddings
        d_x = [sum(d_h1[o] * W1[i][o] for o in range(h1_size))
               for i in range(2 * emb_dim)]
        for i in range(emb_dim):
            E_user[u][i] -= lr_ncf * d_x[i]
            E_item[j][i] -= lr_ncf * d_x[emb_dim + i]

    if (epoch + 1) % 25 == 0:
        rmse = (total_loss / len(obs)) ** 0.5
        print(f"Epoch {epoch+1:3d}: RMSE = {rmse:.4f}")
