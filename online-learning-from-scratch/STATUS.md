# online-learning-from-scratch — Homework Status

## Current Phase: VERIFY complete

### Scripts verified (all 6 pass)
1. `weighted_majority.py` — 37 mistakes, best 35, regret 2 ✓
2. `multiplicative_weights.py` — regret 1.4, well within bound 40.1 ✓
3. `online_gd.py` — ||w-w*||=0.046, w converges ✓
4. `ftrl_comparison.py` — MW regret 16.6, OGD regret 1.7, both sublinear ✓
5. `online_to_batch.py` — last 0.948, averaged 0.948, optimal 0.938 ✓
6. `adagrad.py` — 54.3% improvement, correct weights recovered ✓

### Blog updates
- Comment 1: ~42/~30/~12 → 37/35/2
- Comment 4: "MW adapts faster" → "both achieve sublinear regret"
- Comment 5: "averaging → better generalization" → "both ~94.8%"
- Verified Code badge added
