"""Robustness Comparison — byte-level vs subword under typos.

Code Block 6: Shows how typos destabilize subword tokenization but barely affect bytes.
"""

def char_edit_distance(a, b):
    """Simple Levenshtein distance."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[n][m]

original = "The transformer architecture"
perturbed = "Teh transfomer achitecture"  # 3 typos

# Byte-level: edit distance in byte space
orig_bytes = list(original.encode('utf-8'))
pert_bytes = list(perturbed.encode('utf-8'))
byte_dist = char_edit_distance(orig_bytes, pert_bytes)

# Subword-level: simulated tokens change drastically
# "transformer" -> 1 token; "transfomer" -> 2 tokens (split differently)
orig_tokens = ["The", " transform", "er", " architecture"]     # 4 tokens
pert_tokens = ["T", "eh", " trans", "f", "omer", " ach", "ite", "cture"]  # 8 tokens
token_dist = char_edit_distance(orig_tokens, pert_tokens)

print(f"Original:  '{original}'")
print(f"Perturbed: '{perturbed}'")
print(f"Byte edit distance:    {byte_dist} (small, proportional to typos)")
print(f"Token edit distance:   {token_dist} (large, tokenization shatters)")
print(f"Byte stability ratio:  {byte_dist / len(orig_bytes):.2%}")
print(f"Token stability ratio: {token_dist / len(orig_tokens):.2%}")
