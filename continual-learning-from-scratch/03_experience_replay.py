"""Block 3: Experience Replay — reservoir sampling + replay buffer prevents forgetting."""
import numpy as np
from shared import make_task, sigmoid, train_mlp, accuracy

def reservoir_update(buffer_X, buffer_y, new_X, new_y, buf_size, count):
    """Reservoir sampling: maintain a fixed-size buffer with equal probability."""
    for i in range(len(new_X)):
        count += 1
        if len(buffer_X) < buf_size:
            buffer_X.append(new_X[i])
            buffer_y.append(new_y[i])
        else:
            j = np.random.randint(0, count)
            if j < buf_size:
                buffer_X[j] = new_X[i]
                buffer_y[j] = new_y[i]
    return buffer_X, buffer_y, count

def train_with_replay(X_new, y_new, W1, b1, W2, b2, buffer_X, buffer_y,
                      replay_ratio=0.5, epochs=500, lr=0.05):
    """Train mixing new data with replayed buffer examples."""
    for _ in range(epochs):
        if len(buffer_X) > 0:
            n_replay = max(1, int(len(X_new) * replay_ratio))
            idx = np.random.choice(len(buffer_X), min(n_replay, len(buffer_X)))
            buf_X = np.array([buffer_X[i] for i in idx])
            buf_y = np.array([buffer_y[i] for i in idx])
            X_mix = np.vstack([X_new, buf_X])
            y_mix = np.concatenate([y_new, buf_y])
        else:
            X_mix, y_mix = X_new, y_new
        h = np.maximum(0, X_mix @ W1 + b1)
        out = sigmoid(h @ W2 + b2)
        err = out.ravel() - y_mix
        dW2 = h.T @ err.reshape(-1,1) / len(y_mix)
        db2 = err.mean()
        dh = err.reshape(-1,1) * W2.T * (h > 0)
        dW1 = X_mix.T @ dh / len(y_mix)
        db1 = dh.mean(axis=0)
        W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
    return W1, b1, W2, b2

if __name__ == "__main__":
    X1, y1 = make_task([-1, 0], [1, 0], seed=42)
    X2, y2 = make_task([0, -1], [0, 1], seed=99)

    rng = np.random.RandomState(0)
    W1 = rng.randn(2,8)*0.3; b1 = np.zeros(8)
    W2 = rng.randn(8,1)*0.3; b2 = np.zeros(1)
    W1, b1, W2, b2 = train_mlp(X1, y1, W1, b1, W2, b2)

    buffer_X, buffer_y, count = [], [], 0
    buffer_X, buffer_y, count = reservoir_update(
        buffer_X, buffer_y, X1, y1, buf_size=50, count=count)

    W1, b1, W2, b2 = train_with_replay(X2, y2, W1, b1, W2, b2,
                                         buffer_X, buffer_y, replay_ratio=0.5)
    print(f"Replay: acc_task1={accuracy(X1,y1,W1,b1,W2,b2):.0%}, "
          f"acc_task2={accuracy(X2,y2,W1,b1,W2,b2):.0%}")
    # Expected: Replay: acc_task1=88%, acc_task2=97%
