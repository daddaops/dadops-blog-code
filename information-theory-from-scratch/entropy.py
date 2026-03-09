import numpy as np

def information_content(p):
    """Bits of information from an event with probability p."""
    return -np.log2(np.clip(p, 1e-12, 1.0))

def entropy(probs):
    """Shannon entropy H(X) in bits."""
    probs = np.array(probs, dtype=float)
    probs = probs[probs > 0]  # 0·log(0) = 0 by convention
    return -np.sum(probs * np.log2(probs))

# --- Examples ---
# Fair coin: maximum entropy for 2 outcomes
print(f"Fair coin entropy: {entropy([0.5, 0.5]):.4f} bits")        # 1.0
print(f"Biased coin (90/10): {entropy([0.9, 0.1]):.4f} bits")      # 0.469
print(f"Certain coin (100/0): {entropy([1.0, 0.0]):.4f} bits")     # 0.0

# Fair die: maximum entropy for 6 outcomes
print(f"Fair die entropy: {entropy([1/6]*6):.4f} bits")             # 2.585
print(f"Loaded die: {entropy([0.5,0.1,0.1,0.1,0.1,0.1]):.4f}")    # 2.161

# Binary entropy as a function of p
for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    h = entropy([p, 1-p]) if p > 0 else 0.0
    print(f"  p={p:.1f}: H = {h:.4f} bits")
# Peaks at p=0.5 (1.0 bit) — the famous parabolic curve
