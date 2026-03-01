"""
Code Block 6: Hyperparameter search strategies.

From: https://dadops.dev/blog/ml-experiment-tracking/

SearchSpace defines parameter samplers (log_uniform, choice, uniform).
grid_search() exhaustively tries all combinations.
random_search() samples randomly (Bergstra & Bengio 2012).
successive_halving() prunes worst configs iteratively — O(n log n).

No external dependencies required.
"""

import random
import math


class SearchSpace:
    """Define the space of hyperparameters to explore."""
    @staticmethod
    def log_uniform(low, high):
        return lambda: math.exp(random.uniform(math.log(low), math.log(high)))

    @staticmethod
    def choice(options):
        return lambda: random.choice(options)

    @staticmethod
    def uniform(low, high):
        return lambda: random.uniform(low, high)


def grid_search(param_grid):
    """Exhaustive: try every combination. O(k^d) evaluations."""
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def random_search(search_space, n_trials):
    """Sample randomly: Bergstra & Bengio (2012) showed this
    beats grid search because it explores each dimension independently."""
    for _ in range(n_trials):
        yield {name: sampler() for name, sampler in search_space.items()}


def successive_halving(search_space, n_configs, min_budget, max_budget, evaluate_fn):
    """Start many configs with small budget, prune worst half,
    double budget for survivors. Reaches optimal config in O(n log n)."""
    configs = list(random_search(search_space, n_configs))
    budget = min_budget
    while len(configs) > 1 and budget <= max_budget:
        # Evaluate all surviving configs with current budget
        results = [(evaluate_fn(cfg, budget), cfg) for cfg in configs]
        results.sort(reverse=True)
        # Keep top half
        configs = [cfg for _, cfg in results[:max(len(results) // 2, 1)]]
        budget *= 2
    return configs[0] if configs else None


if __name__ == "__main__":
    print("=== Hyperparameter Search Demo ===\n")
    random.seed(42)

    # Example: grid search
    grid = {
        "lr": [0.001, 0.01, 0.1],
        "batch_size": [32, 64, 128],
    }
    grid_configs = list(grid_search(grid))
    print(f"Grid search: {len(grid_configs)} configs")
    for cfg in grid_configs[:5]:
        print(f"  {cfg}")
    print(f"  ... ({len(grid_configs)} total)")

    # Example: random search
    space = {
        "lr": SearchSpace.log_uniform(1e-5, 1e-1),
        "batch_size": SearchSpace.choice([16, 32, 64, 128]),
        "dropout": SearchSpace.uniform(0.0, 0.5),
    }
    random_configs = list(random_search(space, 10))
    print(f"\nRandom search: 10 configs")
    for cfg in random_configs[:3]:
        print(f"  lr={cfg['lr']:.6f}, batch={cfg['batch_size']}, drop={cfg['dropout']:.3f}")

    # Blog claim: grid 10x10 = 100 evals but only 10 distinct LR values
    grid_10x10 = {
        "lr": [0.001 * i for i in range(1, 11)],
        "batch_size": [8 * i for i in range(1, 11)],
    }
    all_grid = list(grid_search(grid_10x10))
    distinct_lrs = len(set(c["lr"] for c in all_grid))
    print(f"\n10x10 grid: {len(all_grid)} evaluations, {distinct_lrs} distinct LRs")

    # Random search: 100 evals = 100 distinct LR values
    random_100 = list(random_search(
        {"lr": SearchSpace.uniform(0.001, 0.01), "batch_size": SearchSpace.choice(range(8, 81))},
        100
    ))
    distinct_random_lrs = len(set(c["lr"] for c in random_100))
    print(f"Random 100: {len(random_100)} evaluations, {distinct_random_lrs} distinct LRs")

    # Example: successive halving
    def mock_evaluate(cfg, budget):
        # Score peaks near lr=0.003, batch=64
        lr_score = 1.0 - abs(math.log10(cfg["lr"]) - math.log10(0.003))
        bs_score = 1.0 - abs(cfg["batch_size"] - 64) / 128
        return lr_score + bs_score + random.gauss(0, 0.1)

    best = successive_halving(space, n_configs=27, min_budget=1, max_budget=81,
                              evaluate_fn=mock_evaluate)
    print(f"\nSuccessive halving best config:")
    print(f"  lr={best['lr']:.6f}, batch={best['batch_size']}, drop={best['dropout']:.3f}")
