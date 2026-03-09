import random
import math

def rastrigin(x):
    """Rastrigin function: many local optima, one global optimum at x=0."""
    return -(x**2 - 10 * math.cos(2 * math.pi * x) + 10)

def roulette_select(population, scores):
    """Fitness-proportionate selection."""
    min_s = min(scores)
    weights = [s - min_s + 1e-6 for s in scores]
    return random.choices(population, weights=weights)[0]

def tournament_select(population, scores, k=3):
    """Tournament selection with configurable pressure."""
    contestants = random.sample(list(zip(population, scores)), k)
    return max(contestants, key=lambda pair: pair[1])[0]

def rank_select(population, scores):
    """Rank-based selection."""
    ranked = sorted(zip(population, scores), key=lambda p: p[1])
    ranks = list(range(1, len(ranked) + 1))
    return random.choices([ind for ind, _ in ranked], weights=ranks)[0]

def run_ga(select_fn, label, pop_size=60, gens=80):
    random.seed(7)
    pop = [random.uniform(-5.12, 5.12) for _ in range(pop_size)]
    best_history = []
    for gen in range(gens):
        scores = [rastrigin(x) for x in pop]
        best_history.append(max(scores))
        next_gen = []
        for _ in range(pop_size):
            p1 = select_fn(pop, scores)
            p2 = select_fn(pop, scores)
            child = (p1 + p2) / 2 + random.gauss(0, 0.3)
            child = max(-5.12, min(5.12, child))
            next_gen.append(child)
        pop = next_gen
    print(f"{label}: best = {max(best_history):.4f} (gen {best_history.index(max(best_history))})")

if __name__ == "__main__":
    run_ga(roulette_select, "Roulette ")
    run_ga(tournament_select, "Tournament")
    run_ga(rank_select, "Rank      ")
