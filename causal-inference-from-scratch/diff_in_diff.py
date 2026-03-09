"""Difference-in-Differences (DiD) estimation from scratch.

Simulates a company rolling out a new ML recommendation model to 25 cities
(treated) but not 25 others (control). True treatment effect is $8K revenue boost.
DiD correctly removes the baseline gap between groups.
"""
import numpy as np

np.random.seed(42)
n_cities = 50
n_periods = 8  # 4 pre-treatment + 4 post-treatment
treatment_start = 4

# City-level panel data
treated_cities = np.arange(n_cities // 2)
control_cities = np.arange(n_cities // 2, n_cities)

# Baseline revenue differs between groups (treated cities are bigger)
baseline = np.zeros(n_cities)
baseline[treated_cities] = 100  # treated cities start higher
baseline[control_cities] = 80   # control cities start lower

# Common time trend (both groups grow similarly)
time_trend = np.arange(n_periods) * 2.0  # $2K/period growth

# True treatment effect: $8K revenue boost after rollout
true_effect = 8.0

# Generate panel data
revenue = np.zeros((n_cities, n_periods))
for t in range(n_periods):
    noise = np.random.normal(0, 3, n_cities)
    revenue[:, t] = baseline + time_trend[t] + noise
    if t >= treatment_start:
        revenue[treated_cities, t] += true_effect

# DiD estimation
treat_before = revenue[treated_cities, :treatment_start].mean()
treat_after = revenue[treated_cities, treatment_start:].mean()
ctrl_before = revenue[control_cities, :treatment_start].mean()
ctrl_after = revenue[control_cities, treatment_start:].mean()

did_estimate = (treat_after - treat_before) - (ctrl_after - ctrl_before)

print(f"=== Difference-in-Differences ===")
print(f"Treated cities:  before={treat_before:.1f}  after={treat_after:.1f}  change={treat_after-treat_before:+.1f}")
print(f"Control cities:  before={ctrl_before:.1f}  after={ctrl_after:.1f}  change={ctrl_after-ctrl_before:+.1f}")
print(f"\nFirst diff (treated):  {treat_after - treat_before:+.1f}")
print(f"First diff (control):  {ctrl_after - ctrl_before:+.1f}")
print(f"DiD estimate:          {did_estimate:+.1f}")
print(f"True effect:           {true_effect:+.1f}")

# Naive comparison (ignoring pre-treatment differences)
naive = treat_after - ctrl_after
print(f"\nNaive post-only:       {naive:+.1f}  (inflated by baseline gap!)")
print("DiD removes the baseline difference between groups.")

# Check parallel trends assumption (pre-treatment periods)
pre_trends_treat = [revenue[treated_cities, t].mean() for t in range(treatment_start)]
pre_trends_ctrl = [revenue[control_cities, t].mean() for t in range(treatment_start)]
gaps = [pre_trends_treat[t] - pre_trends_ctrl[t] for t in range(treatment_start)]
print(f"\nParallel trends check (pre-treatment gaps):")
print(f"  Gaps: {['%.1f' % g for g in gaps]}")
print(f"  Stable gap ≈ {np.mean(gaps):.1f} confirms parallel trends assumption")
