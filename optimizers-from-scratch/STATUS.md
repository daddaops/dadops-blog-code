# optimizers-from-scratch — Homework Status

## Current Phase: VERIFY complete

### Scripts verified (all 4 pass)
1. `sgd_bowl.py` — Steps 0/1/2/14 all match blog exactly ✓
2. `sgd_ravine.py` — Minor numeric corrections (e.g., 52.3944→52.2927) ✓
3. `momentum_ravine.py` — Major correction: blog fabricated smooth convergence, actual shows overshoot with β=0.9 ✓
4. `optimizer_comparison.py` — Major correction: blog fabricated Adam→0.000001, actual shows all optimizers trapped at ~0.65-0.87 ✓

### Blog updates
- SGD ravine: updated 6 output values
- Momentum ravine: rewrote output + prose (overshoot behavior)
- Optimizer comparison: rewrote output + prose + table (all trapped in local basin)
- Verified Code badge added
