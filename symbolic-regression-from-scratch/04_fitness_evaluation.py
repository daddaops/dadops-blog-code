"""Fitness evaluation with parsimony pressure and Pareto front."""
import numpy as np
from node import Node
from tree_gen import init_population


def evaluate_fitness(tree, x_data, y_data, alpha=0.001):
    """MSE + parsimony pressure. Lower is better."""
    try:
        y_pred = tree.evaluate(x_data)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return 1e12
        mse = np.mean((y_pred - y_data) ** 2)
    except Exception:
        return 1e12
    return mse + alpha * tree.size()


def pareto_front(population, fitnesses):
    """Return indices of Pareto-optimal individuals (MSE vs size)."""
    mses, sizes = [], []
    for i, tree in enumerate(population):
        try:
            mse_only = fitnesses[i] - 0.001 * tree.size()
        except Exception:
            mse_only = 1e12
        mses.append(mse_only)
        sizes.append(tree.size())
    front = []
    for i in range(len(population)):
        dominated = False
        for j in range(len(population)):
            if i == j:
                continue
            if mses[j] <= mses[i] and sizes[j] <= sizes[i]:
                if mses[j] < mses[i] or sizes[j] < sizes[i]:
                    dominated = True
                    break
        if not dominated:
            front.append(i)
    return front


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    x = np.linspace(-3, 3, 50)
    y = x ** 2 + 1.0  # target: x^2 + 1

    pop = init_population(20, 5, rng)
    fitnesses = [evaluate_fitness(t, x, y) for t in pop]

    print("Fitness evaluation on f(x) = x^2 + 1:")
    for i, (tree, fit) in enumerate(zip(pop, fitnesses)):
        if fit < 1e10:
            print(f"  [{i:2d}] fitness={fit:10.4f}  size={tree.size():2d}  expr={tree}")

    front = pareto_front(pop, fitnesses)
    print(f"\nPareto front ({len(front)} individuals):")
    for idx in front:
        print(f"  [{idx:2d}] fitness={fitnesses[idx]:10.4f}  size={pop[idx].size():2d}  expr={pop[idx]}")
