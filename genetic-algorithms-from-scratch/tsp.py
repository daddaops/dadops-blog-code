import random
import math

def tour_distance(tour, cities):
    """Total distance of a tour (returns to start)."""
    dist = 0
    for i in range(len(tour)):
        c1, c2 = cities[tour[i]], cities[tour[(i + 1) % len(tour)]]
        dist += math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    return dist

def order_crossover(p1, p2):
    """OX: copy a segment from p1, fill remaining from p2 in order."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b] = p1[a:b]
    segment = set(p1[a:b])
    fill = [c for c in p2 if c not in segment]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child

def swap_mutate(tour, rate=0.05):
    """Swap two random cities with given probability."""
    if random.random() < rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

if __name__ == "__main__":
    # Generate 20 random cities
    random.seed(42)
    n_cities = 20
    cities = [(random.uniform(0, 100), random.uniform(0, 100))
              for _ in range(n_cities)]

    # GA for TSP
    pop_size, generations = 100, 200
    population = [random.sample(range(n_cities), n_cities)
                  for _ in range(pop_size)]

    best_ever = None
    best_dist = float('inf')

    for gen in range(generations):
        distances = [tour_distance(t, cities) for t in population]

        # Track best
        gen_best_idx = min(range(pop_size), key=lambda i: distances[i])
        if distances[gen_best_idx] < best_dist:
            best_dist = distances[gen_best_idx]
            best_ever = list(population[gen_best_idx])

        # Tournament selection + elitism
        next_gen = [list(best_ever)]  # elitism: keep the best
        while len(next_gen) < pop_size:
            # Tournament select two parents (lower distance = better)
            t1 = min(random.sample(range(pop_size), 3), key=lambda i: distances[i])
            t2 = min(random.sample(range(pop_size), 3), key=lambda i: distances[i])
            child = order_crossover(population[t1], population[t2])
            child = swap_mutate(child, rate=0.15)
            next_gen.append(child)
        population = next_gen

    initial_random = tour_distance(random.sample(range(n_cities), n_cities), cities)
    print(f"Random tour distance: {initial_random:.1f}")
    print(f"GA best distance:     {best_dist:.1f}")
    print(f"Improvement:          {(1 - best_dist / initial_random) * 100:.1f}%")
