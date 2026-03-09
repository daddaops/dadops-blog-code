import random

def single_point_crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:]

def two_point_crossover(p1, p2):
    a, b = sorted(random.sample(range(1, len(p1)), 2))
    return p1[:a] + p2[a:b] + p1[b:]

def uniform_crossover(p1, p2):
    return [random.choice([g1, g2]) for g1, g2 in zip(p1, p2)]

def gaussian_mutate(individual, rate=0.1, sigma=0.3):
    """Mutate real-valued genes with Gaussian noise."""
    return [
        gene + random.gauss(0, sigma) if random.random() < rate else gene
        for gene in individual
    ]

if __name__ == "__main__":
    # Example: apply each operator to a sample chromosome
    random.seed(0)
    a, b = [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]
    print("Single-pt:", single_point_crossover(a, b))
    print("Two-pt:   ", two_point_crossover(a, b))
    print("Uniform:  ", uniform_crossover(a, b))
    print("Mutated:  ", gaussian_mutate(a, rate=0.5, sigma=1.0))

    # Compare crossover vs mutation on f(x, y) = -(x^2 + y^2)
    def sphere_fitness(ind):
        return -(ind[0]**2 + ind[1]**2)  # optimum at (0, 0)

    random.seed(42)
    configs = {
        "Crossover only": (True, False),
        "Mutation only":  (False, True),
        "Both":           (True, True),
    }

    for label, (use_cross, use_mut) in configs.items():
        pop = [[random.uniform(-5, 5), random.uniform(-5, 5)] for _ in range(50)]
        for gen in range(100):
            scores = [sphere_fitness(ind) for ind in pop]
            ranked = sorted(zip(pop, scores), key=lambda p: p[1], reverse=True)
            elite = [ind for ind, _ in ranked[:10]]  # top 20% survive
            next_gen = list(elite)
            while len(next_gen) < 50:
                p1, p2 = random.sample(elite, 2)
                if use_cross:
                    child = [(g1 + g2) / 2 for g1, g2 in zip(p1, p2)]
                else:
                    child = list(random.choice([p1, p2]))
                if use_mut:
                    child = gaussian_mutate(child, rate=0.3, sigma=0.5)
                next_gen.append(child)
            pop = next_gen
        best = max(pop, key=sphere_fitness)
        print(f"{label:<16s}: best = ({best[0]:+.4f}, {best[1]:+.4f}), "
              f"fitness = {sphere_fitness(best):.6f}")
