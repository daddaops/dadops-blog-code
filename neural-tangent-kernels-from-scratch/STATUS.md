# neural-tangent-kernels-from-scratch — Homework Status

## Current Phase: UPDATE complete

### Scripts extracted (5 scripts)
1. `empirical_ntk.py` — NTK via Jacobian
2. `ntk_stability.py` — Width convergence
3. `training_dynamics.py` — GD vs NTK prediction
4. `rich_vs_lazy.py` — Rich/lazy regimes
5. `analytic_ntk.py` — Infinite-width arccosine kernel

### Blog updates
- ntk_stability output: 0.5823/0.2741/0.1284/0.0594 → 0.7243/0.2934/0.1976/0.0991
- Prose "6%" → "10%" for width-1000 variation
- rich_vs_lazy output: all widths now correctly show LAZY (NTK param inherently suppresses feature learning)
- Prose rewritten: explains NTK param → all lazy, rich regime requires μP parameterization
