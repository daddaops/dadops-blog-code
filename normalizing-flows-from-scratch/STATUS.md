# normalizing-flows-from-scratch — Homework Status

## Current Phase: VERIFY complete

### Scripts verified (all 6 pass)
1. `change_of_variables.py` — z=1.281, p=0.5310 ✓
2. `planar_flow.py` — x=[-2.1,2.1], y=[-5.5,6.4], mean=-3.01 ✓
3. `affine_coupling.py` — error=4.44e-16, log_det=-0.1929 ✓
4. `train_realnvp.py` — loss 2.562→2.303 ✓
5. `invertible_conv.py` — log_det=0.0000 (QR orthogonal) ✓
6. `maf.py` — loss 2.496→2.269 ✓

### Blog updates
- All 6 output comments corrected
- Training lr reduced 0.003→0.001 for stability
- Verified Code badge added
