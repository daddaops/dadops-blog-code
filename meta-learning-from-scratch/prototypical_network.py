"""Prototypical Network with episodic training and full backprop.

Combines the PrototypicalNetwork class definition (blog block 2)
and the episode training loop (blog block 3) into a single runnable script.
Implements the gradient computation that the blog omits "for clarity".
"""
import numpy as np


class PrototypicalNetwork:
    """Few-shot classifier via nearest-centroid in learned embedding space."""

    def __init__(self, input_dim, embed_dim=32, hidden_dim=64):
        # Embedding network: input -> hidden -> embed
        # He initialization
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim) * scale2
        self.b2 = np.zeros(embed_dim)

    def embed(self, X):
        """Map inputs to embedding space."""
        h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2               # linear output

    def compute_prototypes(self, support_X, support_y, n_classes):
        """Compute class prototypes as mean embeddings."""
        embeddings = self.embed(support_X)
        prototypes = np.zeros((n_classes, embeddings.shape[1]))
        for c in range(n_classes):
            mask = support_y == c
            prototypes[c] = embeddings[mask].mean(axis=0)
        return prototypes

    def classify(self, query_X, prototypes):
        """Classify queries by nearest prototype (softmin of distances)."""
        q_embed = self.embed(query_X)
        # Negative squared Euclidean distance to each prototype
        # p(y=k|x) = softmax(-||f(x) - c_k||^2)
        dists = -np.sum((q_embed[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
        exp_d = np.exp(dists - dists.max(axis=1, keepdims=True))
        return exp_d / exp_d.sum(axis=1, keepdims=True)  # [n_query, n_classes]

    def episode_forward_backward(self, sX, sy, qX, qy, n_way):
        """Full forward + backward pass through an episode with gradient computation."""
        # --- Forward pass with saved intermediates ---
        # Embed support set
        s_h1_pre = sX @ self.W1 + self.b1
        s_h1 = np.maximum(0, s_h1_pre)
        s_embed = s_h1 @ self.W2 + self.b2

        # Compute prototypes (mean per class)
        prototypes = np.zeros((n_way, s_embed.shape[1]))
        class_counts = np.zeros(n_way)
        for c in range(n_way):
            mask = sy == c
            prototypes[c] = s_embed[mask].mean(axis=0)
            class_counts[c] = mask.sum()

        # Embed query set
        q_h1_pre = qX @ self.W1 + self.b1
        q_h1 = np.maximum(0, q_h1_pre)
        q_embed = q_h1 @ self.W2 + self.b2

        # Compute distances and probabilities
        diff = q_embed[:, None, :] - prototypes[None, :, :]  # [nq, n_way, embed_dim]
        neg_dists = -np.sum(diff ** 2, axis=2)  # [nq, n_way]
        exp_d = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
        probs = exp_d / exp_d.sum(axis=1, keepdims=True)

        # Cross-entropy loss
        nq = len(qy)
        loss = -np.log(probs[range(nq), qy] + 1e-8).mean()
        acc = (probs.argmax(axis=1) == qy).mean()

        # --- Backward pass ---
        # Gradient of cross-entropy w.r.t. neg_dists (softmax + CE)
        d_neg_dists = probs.copy()
        d_neg_dists[range(nq), qy] -= 1
        d_neg_dists /= nq  # [nq, n_way]

        # Gradient of neg_dists w.r.t. diff: neg_dists = -sum(diff^2)
        # d_diff = d_neg_dists * (-2 * diff)
        d_diff = -2 * d_neg_dists[:, :, None] * diff  # [nq, n_way, embed_dim]

        # Gradient w.r.t. query embeddings
        d_q_embed = d_diff.sum(axis=1)  # [nq, embed_dim]

        # Gradient w.r.t. prototypes
        d_prototypes = -d_diff.sum(axis=0)  # [n_way, embed_dim]

        # Backprop through query embedding network
        d_q_h1 = (d_q_embed @ self.W2.T) * (q_h1_pre > 0)
        dW2_q = q_h1.T @ d_q_embed
        db2_q = d_q_embed.sum(axis=0)
        dW1_q = qX.T @ d_q_h1
        db1_q = d_q_h1.sum(axis=0)

        # Backprop through prototype mean -> support embeddings
        d_s_embed = np.zeros_like(s_embed)
        for c in range(n_way):
            mask = sy == c
            d_s_embed[mask] = d_prototypes[c] / class_counts[c]

        # Backprop through support embedding network
        d_s_h1 = (d_s_embed @ self.W2.T) * (s_h1_pre > 0)
        dW2_s = s_h1.T @ d_s_embed
        db2_s = d_s_embed.sum(axis=0)
        dW1_s = sX.T @ d_s_h1
        db1_s = d_s_h1.sum(axis=0)

        # Total gradients (shared weights)
        grads = {
            'W1': dW1_q + dW1_s,
            'b1': db1_q + db1_s,
            'W2': dW2_q + dW2_s,
            'b2': db2_q + db2_s,
        }

        return loss, acc, probs, grads


def sample_episode(data, labels, n_way, k_shot, n_query, rng):
    """Sample one N-way K-shot episode from a pool of classes."""
    unique_classes = np.unique(labels)
    chosen = rng.choice(unique_classes, n_way, replace=False)

    support_X, support_y, query_X, query_y = [], [], [], []
    for new_label, cls in enumerate(chosen):
        cls_indices = np.where(labels == cls)[0]
        selected = rng.choice(cls_indices, k_shot + n_query, replace=False)
        for idx in selected[:k_shot]:
            support_X.append(data[idx])
            support_y.append(new_label)
        for idx in selected[k_shot:]:
            query_X.append(data[idx])
            query_y.append(new_label)
    return (np.array(support_X), np.array(support_y),
            np.array(query_X), np.array(query_y))


def train_prototypical_net(model, data, labels, episodes=1000,
                           n_way=5, k_shot=5, n_query=10, lr=0.001):
    """Train embedding network via episodic meta-learning."""
    rng = np.random.RandomState(42)

    for ep in range(episodes):
        sX, sy, qX, qy = sample_episode(data, labels, n_way, k_shot, n_query, rng)

        # Forward + backward with full gradient computation
        loss, acc, probs, grads = model.episode_forward_backward(sX, sy, qX, qy, n_way)

        # SGD update
        model.W1 -= lr * grads['W1']
        model.b1 -= lr * grads['b1']
        model.W2 -= lr * grads['W2']
        model.b2 -= lr * grads['b2']

        if ep % 200 == 0:
            print(f"Episode {ep:4d}: loss={loss:.3f}, query_acc={acc:.0%}")


if __name__ == "__main__":
    # Generate synthetic meta-learning dataset: 20 classes, 30 examples each
    np.random.seed(0)
    n_meta_classes = 20
    n_examples_per_class = 30
    dim = 20

    # Each class is a cluster centered at a random point
    data, labels = [], []
    for c in range(n_meta_classes):
        center = np.random.randn(dim) * 1.0
        for _ in range(n_examples_per_class):
            data.append(center + np.random.randn(dim) * 1.0)
            labels.append(c)
    data, labels = np.array(data), np.array(labels)

    model = PrototypicalNetwork(input_dim=dim, embed_dim=32, hidden_dim=64)
    train_prototypical_net(model, data, labels, episodes=1000,
                           n_way=5, k_shot=5, n_query=10, lr=0.01)
