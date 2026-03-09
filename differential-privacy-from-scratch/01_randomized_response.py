import numpy as np

def randomized_response(true_answer):
    """Classic randomized response (Warner 1965).
    Flip coin 1: heads (p=0.5) -> answer truthfully.
    Tails -> flip coin 2, report that coin instead."""
    if np.random.random() < 0.5:
        return true_answer       # truth
    else:
        return np.random.random() < 0.5  # random coin

# Simulate: 1000 people, 40% truly answer "Yes"
np.random.seed(42)
n = 1000
true_rate = 0.40
true_answers = np.random.random(n) < true_rate
noisy = [randomized_response(a) for a in true_answers]
observed_yes = sum(noisy) / n

# Recover the true rate from noisy responses:
# E[observed] = 0.5 * true_rate + 0.5 * 0.5
#             = 0.5 * true_rate + 0.25
estimated_rate = (observed_yes - 0.25) / 0.5

print(f"True rate:      {true_rate:.3f}")
print(f"Observed rate:  {observed_yes:.3f}")
print(f"Estimated rate: {estimated_rate:.3f}")
# Privacy: max ratio = P(Yes|true=Yes)/P(Yes|true=No)
#        = 0.75 / 0.25 = 3, so epsilon = ln(3) ~ 1.10
