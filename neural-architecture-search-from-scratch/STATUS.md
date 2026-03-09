# neural-architecture-search-from-scratch — Homework Status

## Current Phase: RUN complete

### Scripts verified (all 6 pass)
1. `search_space.py` — 5,625,000,000 ✓ matches blog
2. `successive_halving.py` — best=0.585/0.569/0.559/0.592 (blog: 0.812/0.791/0.815/0.836)
3. `evolutionary_search.py` — best=0.739, 0.677→0.739 (blog: 0.858, 0.777→0.858)
4. `darts_search.py` — architecture matches ✓, edge 0 prob 0.902 (blog: 0.981)
5. `supernet.py` — all rankings ≈0.500 (blog: 0.923/0.867/0.845/0.712/0.534), updates=8000 ✓
6. `hardware_aware.py` — values differ from blog

### Discrepancies found
- successive_halving: all round-best scores fabricated (~0.8 vs actual ~0.5-0.6)
- evolutionary: best fitness 0.858→0.739, improvement trajectory differs
- darts: edge 0 probability 0.981→0.902
- supernet: rankings completely fabricated — multiplicative forward pass converges all weights near zero, all ops score ~0.500
- hardware_aware: all values differ
