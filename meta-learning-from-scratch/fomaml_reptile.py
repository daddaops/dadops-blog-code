"""FOMAML and Reptile: first-order approximations to MAML.

Compares FOMAML (drop Hessian), Reptile (move toward task solutions),
and MAML on sinusoid regression.
"""
import numpy as np


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
    # Gradient clipping
    for k in grads:
        g_norm = np.linalg.norm(grads[k])
        if g_norm > 10.0:
            grads[k] = grads[k] * 10.0 / g_norm
    return loss, grads


def fomaml_step(theta, task_data, inner_steps=3, inner_lr=0.01):
    """FOMAML: like MAML but drop second-order terms."""
    x_s, y_s, x_q, y_q = task_data
    adapted = {k: v.copy() for k, v in theta.items()}

    # Inner loop — identical to MAML
    for _ in range(inner_steps):
        _, g = mse_grad(adapted, x_s, y_s)
        adapted = {k: adapted[k] - inner_lr * g[k] for k in adapted}

    # Key difference: compute gradient w.r.t. ADAPTED params only
    # No differentiation through the inner loop (no Hessian)
    _, outer_g = mse_grad(adapted, x_q, y_q)
    return outer_g  # gradient at theta', used to update theta


def reptile_step(theta, task_data, inner_steps=5, inner_lr=0.01):
    """Reptile: take SGD steps on task, move toward result."""
    x_s, y_s = task_data[0], task_data[1]
    adapted = {k: v.copy() for k, v in theta.items()}

    # Take T gradient steps on this task
    for _ in range(inner_steps):
        _, g = mse_grad(adapted, x_s, y_s)
        adapted = {k: adapted[k] - inner_lr * g[k] for k in adapted}

    # Meta-update: move initialization toward adapted params
    # theta <- theta + epsilon * (adapted - theta)
    direction = {k: adapted[k] - theta[k] for k in theta}
    return direction  # used as: theta[k] += epsilon * direction[k]


def init_params(rng):
    """Initialize MLP parameters: 1 -> 40 -> 40 -> 1."""
    return {'W1': rng.randn(1, 40)*0.1,   'b1': np.zeros(40),
            'W2': rng.randn(40, 40)*0.05,  'b2': np.zeros(40),
            'W3': rng.randn(40, 1)*0.05,   'b3': np.zeros(1)}


def sample_task(rng, k_support=5, k_query=10):
    """Sample a random sinusoid task."""
    A = rng.uniform(0.5, 5.0)
    phi = rng.uniform(0, np.pi)
    x_all = rng.uniform(-5, 5, (k_support + k_query, 1))
    y_all = A * np.sin(x_all + phi)
    return (x_all[:k_support], y_all[:k_support],
            x_all[k_support:], y_all[k_support:])


def evaluate(theta, rng, n_eval=100, inner_steps=3, inner_lr=0.01):
    """Evaluate meta-learned init on new tasks."""
    losses = []
    for _ in range(n_eval):
        task = sample_task(rng)
        x_s, y_s, x_q, y_q = task
        adapted = {k: v.copy() for k, v in theta.items()}
        for _ in range(inner_steps):
            _, g = mse_grad(adapted, x_s, y_s)
            adapted = {k: adapted[k] - inner_lr * g[k] for k in adapted}
        loss, _ = mse_grad(adapted, x_q, y_q)
        losses.append(loss)
    return np.mean(losses)


if __name__ == "__main__":
    n_tasks = 10000
    outer_lr = 0.001

    for method_name, method_fn, epsilon in [
        ("FOMAML", "fomaml", 0.001),
        ("Reptile", "reptile", 0.001),
    ]:
        rng = np.random.RandomState(42)
        theta = init_params(rng)

        for t in range(n_tasks):
            task = sample_task(rng)

            if method_name == "FOMAML":
                outer_g = fomaml_step(theta, task, inner_steps=3, inner_lr=0.01)
                theta = {k: theta[k] - epsilon * outer_g[k] for k in theta}
            else:
                direction = reptile_step(theta, task, inner_steps=5, inner_lr=0.01)
                theta = {k: theta[k] + epsilon * direction[k] for k in theta}

            if t % 2000 == 0:
                eval_rng = np.random.RandomState(999)
                eval_loss = evaluate(theta, eval_rng)
                print(f"{method_name} Task {t:5d}: eval_loss={eval_loss:.4f}")

        print()
