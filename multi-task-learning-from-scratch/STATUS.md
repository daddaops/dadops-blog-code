# multi-task-learning-from-scratch — Homework Status

## Current Phase: VERIFY complete

### Scripts verified (all 6 pass)
1. `multi_task_net.py` — 1410 params, regression mean=0.5964, cls range=[0.04, 0.78]
2. `uniform_weighting.py` — L_reg=87.4 vs L_cls=0.87, ratio 100x
3. `uncertainty_weighting.py` — Final weights [0.007, 0.803], cls 109x upweighted
4. `pcgrad.py` — Cosine sim=-0.968 (corrected from -0.969)
5. `gradnorm.py` — Final weights [0.100, 2.029], cls upweighted
6. `train_mtl.py` — 5 steps (finite-diff), L_reg=8.01, L_cls=0.74, cos=-0.108

### Blog updates
- PCGrad cosine similarity: -0.969 → -0.968
- Verified Code badge added
