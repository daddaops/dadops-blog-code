# normalization-from-scratch — Homework Status

## Current Phase: VERIFY complete

### Scripts verified (all 7 pass)
1. `gradient_explosion.py` — std grows to 2.9e+11 at layer 50 ✓
2. `batch_norm.py` — mean=-0.0000, std=1.0000 + batch dependence ✓
3. `layer_norm.py` — Identical outputs regardless of batch ✓
4. `rms_norm.py` — BatchNorm/LayerNorm/RMSNorm comparison ✓
5. `residual_comparison.py` — No-norm/post-norm/pre-norm table ✓
6. `performance_comparison.py` — RMSNorm ~2x faster ✓
7. `llama_block.py` — Stable std across 4 blocks ✓

### Blog updates
- No output comments needed correction (all match exactly)
- Verified Code badge added
