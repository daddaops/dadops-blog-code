import random
import math

def rastrigin_2d(x, y):
    """Rastrigin function: global optimum at (0, 0), many local optima."""
    return -(x**2 - 10*math.cos(2*math.pi*x) + y**2 - 10*math.cos(2*math.pi*y) + 20)

def simplified_cma_es(fn, dim=2, pop_size=20, generations=100):
    """Simplified CMA-ES: adapts mean and step size (not full covariance)."""
    random.seed(42)
    mean = [random.uniform(-3, 3) for _ in range(dim)]
    sigma = 1.0  # initial step size

    best_score = float('-inf')
    best_solution = None

    for gen in range(generations):
        # Sample population from N(mean, sigma^2 * I)
        samples = []
        for _ in range(pop_size):
            ind = [mean[d] + sigma * random.gauss(0, 1) for d in range(dim)]
            samples.append(ind)

        # Evaluate and rank
        scored = [(ind, fn(*ind)) for ind in samples]
        scored.sort(key=lambda p: p[1], reverse=True)

        # Track best
        if scored[0][1] > best_score:
            best_score = scored[0][1]
            best_solution = scored[0][0]

        # Update mean: weighted average of top half
        mu = pop_size // 2
        elite = [ind for ind, _ in scored[:mu]]
        new_mean = [sum(e[d] for e in elite) / mu for d in range(dim)]

        # Adapt step size based on improvement
        mean_shift = math.sqrt(sum((new_mean[d] - mean[d])**2 for d in range(dim)))
        expected_shift = sigma * math.sqrt(dim)
        ratio = mean_shift / expected_shift if expected_shift > 0 else 1.0
        sigma *= math.exp(0.2 * (ratio - 1))  # increase if moving fast, decrease if stalled
        sigma = max(1e-8, min(sigma, 5.0))

        mean = new_mean

    return best_solution, best_score

if __name__ == "__main__":
    solution, score = simplified_cma_es(rastrigin_2d)
    print(f"CMA-ES found: ({solution[0]:.6f}, {solution[1]:.6f})")
    print(f"Fitness: {score:.6f} (optimum = 0.0)")
