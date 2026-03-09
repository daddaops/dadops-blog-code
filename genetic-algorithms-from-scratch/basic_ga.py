import random
import math

def fitness(x):
    """Multimodal function with many local optima."""
    return x * math.sin(10 * math.pi * x) + 2.0

def encode(x, bits=16):
    """Encode a float in [0, 2] as a binary string."""
    scaled = int(x / 2.0 * (2**bits - 1))
    return format(scaled, f'0{bits}b')

def decode(bitstring):
    """Decode a binary string back to a float in [0, 2]."""
    return int(bitstring, 2) / (2**len(bitstring) - 1) * 2.0

def crossover(parent1, parent2):
    """Single-point crossover."""
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

def mutate(bitstring, rate=0.05):
    """Bit-flip mutation."""
    return ''.join(
        ('1' if b == '0' else '0') if random.random() < rate else b
        for b in bitstring
    )

if __name__ == "__main__":
    # Initialize population of random binary strings
    random.seed(42)
    pop_size, bits, generations = 40, 16, 50
    population = [encode(random.uniform(0, 2), bits) for _ in range(pop_size)]

    for gen in range(generations):
        scores = [fitness(decode(ind)) for ind in population]
        # Roulette wheel selection (shift scores to be positive)
        min_s = min(scores)
        weights = [s - min_s + 1e-6 for s in scores]
        total = sum(weights)
        probs = [w / total for w in weights]

        next_gen = []
        for _ in range(pop_size):
            p1 = random.choices(population, weights=probs)[0]
            p2 = random.choices(population, weights=probs)[0]
            child = crossover(p1, p2)
            child = mutate(child)
            next_gen.append(child)
        population = next_gen

    best = max(population, key=lambda ind: fitness(decode(ind)))
    print(f"Best x = {decode(best):.4f}, f(x) = {fitness(decode(best)):.4f}")
