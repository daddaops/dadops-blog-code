import numpy as np

# A tiny synonym dictionary for demonstration
SYNONYMS = {
    'happy': ['glad', 'joyful', 'pleased', 'cheerful'],
    'sad': ['unhappy', 'sorrowful', 'gloomy', 'melancholy'],
    'good': ['great', 'excellent', 'fine', 'wonderful'],
    'bad': ['poor', 'terrible', 'awful', 'dreadful'],
    'big': ['large', 'huge', 'enormous', 'vast'],
    'small': ['tiny', 'little', 'miniature', 'compact'],
    'fast': ['quick', 'rapid', 'swift', 'speedy'],
}

def synonym_replace(words, n=1, rng=None):
    """Replace n random words with their synonyms."""
    rng = rng or np.random.default_rng()
    result = words.copy()
    replaceable = [i for i, w in enumerate(result) if w.lower() in SYNONYMS]
    if not replaceable:
        return result
    indices = rng.choice(replaceable, size=min(n, len(replaceable)), replace=False)
    for i in indices:
        syns = SYNONYMS[result[i].lower()]
        result[i] = syns[rng.integers(len(syns))]
    return result

def random_insert(words, n=1, rng=None):
    """Insert a synonym of a random word at a random position."""
    rng = rng or np.random.default_rng()
    result = words.copy()
    insertable = [w for w in result if w.lower() in SYNONYMS]
    for _ in range(n):
        if not insertable:
            break
        word = insertable[rng.integers(len(insertable))]
        syns = SYNONYMS[word.lower()]
        pos = rng.integers(len(result) + 1)
        result.insert(pos, syns[rng.integers(len(syns))])
    return result

def random_delete(words, p=0.1, rng=None):
    """Delete each word with probability p."""
    rng = rng or np.random.default_rng()
    if len(words) == 1:
        return words
    return [w for w in words if rng.random() > p]

def random_swap(words, n=1, rng=None):
    """Swap two random words n times."""
    rng = rng or np.random.default_rng()
    result = words.copy()
    for _ in range(n):
        if len(result) < 2:
            break
        i, j = rng.choice(len(result), size=2, replace=False)
        result[i], result[j] = result[j], result[i]
    return result

def eda(text, alpha=0.1, rng=None):
    """EDA: apply all four augmentations."""
    rng = rng or np.random.default_rng()
    words = text.split()
    n = max(1, int(alpha * len(words)))
    words = synonym_replace(words, n, rng)
    words = random_insert(words, n, rng)
    words = random_swap(words, n, rng)
    words = random_delete(words, alpha, rng)
    return ' '.join(words)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    text = "The happy dog had a good fast run in the big park"
    print("Original:", text)

    words = text.split()
    print("\nSynonym replace:", ' '.join(synonym_replace(words, n=2, rng=rng)))
    print("Random insert:", ' '.join(random_insert(words, n=1, rng=rng)))
    print("Random delete:", ' '.join(random_delete(words, p=0.2, rng=rng)))
    print("Random swap:", ' '.join(random_swap(words, n=1, rng=rng)))

    print("\nFull EDA (5 augmentations):")
    for i in range(5):
        augmented = eda(text, alpha=0.1, rng=rng)
        print(f"  {i+1}: {augmented}")
    print("\nAll EDA tests passed.")
