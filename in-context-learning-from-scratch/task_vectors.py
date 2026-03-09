import numpy as np

def generate_icl_tasks(n_examples=8, n_tasks=4, dim=3, seed=42):
    """Generate simple linear tasks for ICL demonstration."""
    rng = np.random.RandomState(seed)
    tasks = {}
    task_names = ["double", "negate", "shift+3", "halve"]
    task_fns = [
        lambda x: 2 * x,
        lambda x: -x,
        lambda x: x + 3,
        lambda x: x / 2
    ]

    for name, fn in zip(task_names, task_fns):
        X = rng.randn(n_examples, dim)
        y = np.array([fn(x) for x in X])
        tasks[name] = (X, y)

    return tasks

def simulate_task_representations(tasks, hidden_dim=16, seed=42):
    """Simulate extracting 'task vectors' from a transformer."""
    rng = np.random.RandomState(seed)

    W_encode = rng.randn(4, hidden_dim)  # D+1 -> hidden_dim

    representations = {}
    for name, (X, y) in tasks.items():
        tokens = np.column_stack([X, y[:, :1]])  # (n_examples, 4)
        hidden = tokens @ W_encode  # (n_examples, hidden_dim)
        task_vec = hidden.mean(axis=0)
        representations[name] = task_vec

    return representations

def pca_2d(vectors):
    """Project vectors to 2D using PCA."""
    matrix = np.array(list(vectors.values()))
    centered = matrix - matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:2].T
    return {name: projected[i] for i, name in enumerate(vectors)}

# Run the experiment
tasks = generate_icl_tasks()
reps = simulate_task_representations(tasks)
coords = pca_2d(reps)

print("Task vectors in 2D (PCA projection):")
print("-" * 40)
for name, (x, y) in coords.items():
    print(f"  {name:10s}: ({x:+.2f}, {y:+.2f})")

# Show distances between tasks
names = list(coords.keys())
print("\nPairwise distances (similar tasks cluster):")
for i in range(len(names)):
    for j in range(i+1, len(names)):
        d = np.linalg.norm(
            np.array(coords[names[i]]) - np.array(coords[names[j]])
        )
        print(f"  {names[i]:10s} <-> {names[j]:10s}: {d:.3f}")
