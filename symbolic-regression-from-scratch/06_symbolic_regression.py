"""Full symbolic regression with genetic programming."""
import copy
import numpy as np
from node import Node
from tree_gen import FUNCTIONS, random_terminal, grow_tree, init_population
from genetic_ops import (get_random_node, tournament_select,
                         subtree_crossover, point_mutation, hoist_mutation)
from linear_scaling import linear_scale_fitness


def symbolic_regression(x_data, y_data, pop_size=300, generations=50,
                        max_depth=6, tournament_k=5, seed=42):
    rng = np.random.RandomState(seed)
    population = init_population(pop_size, max_depth, rng)

    best_tree, best_fit = None, 1e12
    for gen in range(generations):
        # Evaluate
        fitnesses = [linear_scale_fitness(t, x_data, y_data)
                     for t in population]

        # Track best
        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < best_fit:
            best_fit = fitnesses[gen_best_idx]
            best_tree = copy.deepcopy(population[gen_best_idx])

        # Create next generation
        new_pop = [copy.deepcopy(best_tree)]  # elitism
        while len(new_pop) < pop_size:
            r = rng.random()
            if r < 0.9:  # 90% crossover
                p1 = tournament_select(population, fitnesses, tournament_k, rng)
                p2 = tournament_select(population, fitnesses, tournament_k, rng)
                child = subtree_crossover(p1, p2, rng, max_depth)
            elif r < 0.95:  # 5% point mutation
                p = tournament_select(population, fitnesses, tournament_k, rng)
                child = point_mutation(p, rng)
            else:  # 5% hoist mutation
                p = tournament_select(population, fitnesses, tournament_k, rng)
                child = hoist_mutation(p, rng)
            new_pop.append(child)
        population = new_pop

        if gen % 10 == 0:
            print(f"Gen {gen}: best fitness = {best_fit:.6f}, "
                  f"expr = {best_tree}, size = {best_tree.size()}")

    return best_tree, best_fit


if __name__ == "__main__":
    # Target: f(x) = x^2 + sin(x)
    x = np.linspace(-3, 3, 100)
    y = x ** 2 + np.sin(x)

    print("Symbolic Regression: discovering f(x) = x^2 + sin(x)")
    print("=" * 60)
    best, fit = symbolic_regression(x, y, pop_size=300, generations=50, seed=42)
    print("=" * 60)
    print(f"\nBest expression: {best}")
    print(f"Best fitness: {fit:.6f}")
    print(f"Tree size: {best.size()}")
    print(f"Tree depth: {best.depth()}")
