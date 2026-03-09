import random
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def forward(weights, x1, x2):
    """2-input, 4-hidden, 1-output network. 17 weights total."""
    # Hidden layer: 4 neurons, each with 2 inputs + 1 bias = 12 weights
    hidden = []
    for i in range(4):
        z = weights[i*3] * x1 + weights[i*3+1] * x2 + weights[i*3+2]
        hidden.append(sigmoid(z))
    # Output: 1 neuron with 4 inputs + 1 bias = 5 weights
    z = sum(weights[12+i] * hidden[i] for i in range(4)) + weights[16]
    return sigmoid(z)

def xor_fitness(weights):
    """Fitness = negative total error on all 4 XOR cases."""
    cases = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    error = sum((forward(weights, x1, x2) - y)**2 for x1, x2, y in cases)
    return -error  # higher is better

if __name__ == "__main__":
    # Evolve XOR solver
    random.seed(42)
    n_weights = 17  # 2*4 + 4 biases + 4*1 + 1 bias
    pop_size = 200
    population = [[random.gauss(0, 1) for _ in range(n_weights)]
                  for _ in range(pop_size)]

    for gen in range(150):
        scores = [xor_fitness(ind) for ind in population]

        # Elitism: keep top 20
        ranked = sorted(zip(population, scores), key=lambda p: p[1], reverse=True)
        elite = [list(ind) for ind, _ in ranked[:20]]

        # Breed next generation
        next_gen = [list(e) for e in elite]
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(elite, 2)
            # Uniform crossover for real-valued weights
            child = [random.choice([g1, g2]) for g1, g2 in zip(p1, p2)]
            # Gaussian mutation
            child = [g + random.gauss(0, 0.2) if random.random() < 0.15 else g
                     for g in child]
            next_gen.append(child)
        population = next_gen

    best = max(population, key=xor_fitness)
    print("Evolved XOR network predictions:")
    for x1, x2, expected in [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]:
        pred = forward(best, x1, x2)
        print(f"  ({x1}, {x2}) -> {pred:.4f}  (expected {expected})")
