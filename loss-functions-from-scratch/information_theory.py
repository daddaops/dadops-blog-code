import numpy as np

# Surprise = -log(p)
# Using natural log (nats), but log2 gives bits

print(f"Sun rises (p=0.999):  surprise = {-np.log(0.999):.4f} nats")
print(f"Snow in Miami (p=0.001): surprise = {-np.log(0.001):.4f} nats")
# Sun rises (p=0.999):  surprise = 0.0010 nats
# Snow in Miami (p=0.001): surprise = 6.9078 nats

def entropy(p_dist):
    """Shannon entropy of a probability distribution (in nats)."""
    p_dist = np.array(p_dist)
    p_dist = p_dist[p_dist > 0]  # skip zeros (0*log(0) = 0 by convention)
    return -np.sum(p_dist * np.log(p_dist))

# Fair coin: maximum uncertainty
print(f"Fair coin [0.5, 0.5]:    H = {entropy([0.5, 0.5]):.4f} nats")
# Fair coin [0.5, 0.5]:    H = 0.6931 nats

# Loaded coin: mostly predictable
print(f"Loaded coin [0.99, 0.01]: H = {entropy([0.99, 0.01]):.4f} nats")
# Loaded coin [0.99, 0.01]: H = 0.0560 nats

# Certain outcome: no surprise at all
print(f"Sure thing [1.0, 0.0]:   H = {entropy([1.0, 0.0]):.4f} nats")
# Sure thing [1.0, 0.0]:   H = 0.0000 nats

def cross_entropy(p_true, q_predicted):
    """Cross-entropy H(P, Q): surprise of Q under P's reality."""
    p = np.array(p_true)
    q = np.array(q_predicted)
    q = np.clip(q, 1e-15, 1.0)  # numerical safety
    return -np.sum(p * np.log(q))

# True distribution: 90% sunny, 10% rainy
P = [0.9, 0.1]

# Model A: good forecast (80% sunny, 20% rainy)
Q_good = [0.8, 0.2]

# Model B: bad forecast (50% sunny, 50% rainy)
Q_bad = [0.5, 0.5]

print(f"Entropy of reality H(P):     {entropy(P):.4f} nats")
print(f"Cross-entropy H(P, Q_good):  {cross_entropy(P, Q_good):.4f} nats")
print(f"Cross-entropy H(P, Q_bad):   {cross_entropy(P, Q_bad):.4f} nats")
# Entropy of reality H(P):     0.3251 nats
# Cross-entropy H(P, Q_good):  0.3618 nats  <- close to H(P), a good model!
# Cross-entropy H(P, Q_bad):   0.6931 nats  <- far from H(P), a bad model!

def kl_divergence(p_true, q_predicted):
    """KL(P || Q) = H(P, Q) - H(P)"""
    return cross_entropy(p_true, q_predicted) - entropy(p_true)

print(f"KL(P || Q_good): {kl_divergence(P, Q_good):.4f} nats")
print(f"KL(P || Q_bad):  {kl_divergence(P, Q_bad):.4f} nats")
# KL(P || Q_good): 0.0367 nats  <- small gap, good model
# KL(P || Q_bad):  0.3681 nats  <- large gap, bad model
