"""MAML for few-shot sinusoid regression.

Learns an initialization for a small MLP such that a few gradient
steps on a new sinusoid task (y = A*sin(x + phi)) produces a good fit.
"""
import numpy as np


def maml_sinusoid(n_tasks=10000, inner_steps=3, inner_lr=0.01,
                  outer_lr=0.001, k_support=5, k_query=10):
    """MAML for few-shot sinusoid regression."""
    rng = np.random.RandomState(42)

    # Simple MLP: 1 -> 40 -> 40 -> 1
    def init_params():
        s1, s2, s3 = np.sqrt(2/1), np.sqrt(2/40), np.sqrt(2/40)
        return {'W1': rng.randn(1, 40)*s1,  'b1': np.zeros(40),
                'W2': rng.randn(40, 40)*s2, 'b2': np.zeros(40),
                'W3': rng.randn(40, 1)*s3,  'b3': np.zeros(1)}

    def forward(params, x):
        h1 = np.maximum(0, x @ params['W1'] + params['b1'])
        h2 = np.maximum(0, h1 @ params['W2'] + params['b2'])
        return h2 @ params['W3'] + params['b3']

    def mse_grad(params, x, y):
        """Compute MSE loss and gradients via manual backprop."""
        h1_pre = x @ params['W1'] + params['b1']
        h1 = np.maximum(0, h1_pre)
        h2_pre = h1 @ params['W2'] + params['b2']
        h2 = np.maximum(0, h2_pre)
        pred = h2 @ params['W3'] + params['b3']

        loss = ((pred - y) ** 2).mean()
        n = len(x)
        dpred = 2 * (pred - y) / n
        grads = {}
        grads['W3'] = h2.T @ dpred
        grads['b3'] = dpred.sum(axis=0)
        dh2 = dpred @ params['W3'].T * (h2_pre > 0)
        grads['W2'] = h1.T @ dh2
        grads['b2'] = dh2.sum(axis=0)
        dh1 = dh2 @ params['W2'].T * (h1_pre > 0)
        grads['W1'] = x.T @ dh1
        grads['b1'] = dh1.sum(axis=0)
        return loss, grads

    theta = init_params()

    for task in range(n_tasks):
        # Sample random sinusoid: y = A * sin(x + phi)
        A = rng.uniform(0.5, 5.0)
        phi = rng.uniform(0, np.pi)
        x_all = rng.uniform(-5, 5, (k_support + k_query, 1))
        y_all = A * np.sin(x_all + phi)
        x_s, y_s = x_all[:k_support], y_all[:k_support]
        x_q, y_q = x_all[k_support:], y_all[k_support:]

        # Inner loop: adapt theta to this task
        adapted = {k: v.copy() for k, v in theta.items()}
        for _ in range(inner_steps):
            _, g = mse_grad(adapted, x_s, y_s)
            adapted = {k: adapted[k] - inner_lr * g[k] for k in adapted}

        # Outer loop: evaluate adapted params on query, update theta
        loss, outer_g = mse_grad(adapted, x_q, y_q)
        # (Full MAML: outer_g should go through inner loop — simplified here)
        theta = {k: theta[k] - outer_lr * outer_g[k] for k in theta}

        if task % 2000 == 0:
            print(f"Task {task:5d}: query_loss={loss:.4f}")

    return theta


if __name__ == "__main__":
    theta = maml_sinusoid()
