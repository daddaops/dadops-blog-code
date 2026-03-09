# neural-tangent-kernels-from-scratch — Homework Status

## Current Phase: VERIFY complete

### Scripts verified (all 5 pass)
1. `empirical_ntk.py` — NTK shape (20,20), NTK[0,0]=4.0636, PSD=True
2. `ntk_stability.py` — Width 10: 0.7243, 50: 0.2934, 200: 0.1976, 1000: 0.0991
3. `training_dynamics.py` — GD closely matches NTK prediction at all widths
4. `rich_vs_lazy.py` — All widths LAZY under NTK parameterization (correct)
5. `analytic_ntk.py` — Depth 1: 2.660, Depth 2: 2.405, Depth 5: 2.148

### Blog updates
- ntk_stability output: 0.5823/0.2741/0.1284/0.0594 → 0.7243/0.2934/0.1976/0.0991
- Prose "6%" → "10%" for width-1000 variation
- rich_vs_lazy output: all widths now correctly show LAZY (NTK param inherently suppresses feature learning)
- Prose rewritten: explains NTK param → all lazy, rich regime requires μP parameterization
- Verified Code badge added
