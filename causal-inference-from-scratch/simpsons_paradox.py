"""Simpson's Paradox demo using kidney stone treatment data (Charig et al., 1986).

Treatment A wins within each subgroup, but Treatment B wins in aggregate.
Stone size is the confounder: doctors assigned the more aggressive treatment
to harder cases.

Expected output:
  === Stratified by Stone Size ===
  Small stones: A=93.1% vs B=86.7%  (A wins by +6.4%)
  Large stones: A=73.0% vs B=68.8%  (A wins by +4.2%)

  === Aggregate (ignoring stone size) ===
  A=78.0% vs B=82.6%  (B wins!)
"""
import numpy as np

# Kidney stone treatment data (Charig et al., 1986)
# Rows: [successes, total] for Treatment A and B
small_stones = {'A': (81, 87), 'B': (234, 270)}
large_stones = {'A': (192, 263), 'B': (55, 80)}

def success_rate(successes, total):
    return successes / total * 100

# Within each subgroup, Treatment A wins
print("=== Stratified by Stone Size ===")
for name, data in [("Small stones", small_stones), ("Large stones", large_stones)]:
    rate_a = success_rate(*data['A'])
    rate_b = success_rate(*data['B'])
    print(f"{name}: A={rate_a:.1f}% vs B={rate_b:.1f}%  (A wins by {rate_a - rate_b:+.1f}%)")

# But in aggregate, Treatment B wins!
agg_a = (81 + 192, 87 + 263)   # total A
agg_b = (234 + 55, 270 + 80)   # total B
print(f"\n=== Aggregate (ignoring stone size) ===")
print(f"A={success_rate(*agg_a):.1f}% vs B={success_rate(*agg_b):.1f}%  (B wins!)")

# Why? Treatment assignment is confounded by stone size
print(f"\n=== The Confounder ===")
print(f"A treats {263}/{87+263} = {263/350*100:.0f}% large stones (harder cases)")
print(f"B treats {80}/{270+80} = {80/350*100:.0f}% large stones")
print("Doctors assigned the more aggressive treatment to harder cases.")
print("Stone size confounds the treatment-outcome relationship.")
