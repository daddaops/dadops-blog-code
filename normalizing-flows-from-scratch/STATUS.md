# normalizing-flows-from-scratch — Homework Status

## Current Phase: UPDATE complete

### Scripts extracted and run (6 scripts)
1. `change_of_variables.py` — z=1.281, p=0.5310 ✓
2. `planar_flow.py` — x=[-2.1,2.1], y=[-5.5,6.4], mean=-3.01 ✓
3. `affine_coupling.py` — error=4.44e-16, log_det=-0.1929 ✓
4. `train_realnvp.py` — loss 2.562→2.303 (lr=0.001, 600 epochs) ✓
5. `invertible_conv.py` — log_det=0.0000 (QR is orthogonal) ✓
6. `maf.py` — loss 2.496→2.269 (lr=0.001, 400 epochs) ✓

### Blog updates
- change_of_variables: z=1.282→1.281, p=0.5313→0.5310
- planar_flow: range and mean corrected
- affine_coupling: error and log_det corrected
- invertible_conv: log_det 0.1342→0.0000 (orthogonal matrix has unit det)
- train_realnvp: lr 0.003→0.001, loss values corrected
- maf: lr 0.003→0.001, loss values corrected
