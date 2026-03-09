# naive-bayes-from-scratch — Homework Status

## Current Phase: RUN complete

### Scripts verified (all 5 pass)
1. `gaussian_nb.py` — [1 0 1] ✓ matches blog
2. `multinomial_nb.py` — spam, -10.12, -14.71 (blog: -7.51, -11.94)
3. `bernoulli_nb.py` — spam, -11.92, -17.59
4. `correlated_features.py` — 93.3% (blog: 98.3%)
5. `scenario_comparison.py` — 66.7%, 76.7%, 82.2%, 58.3% (blog: 80.0%, 76.7%, 97.8%, 88.3%)

### Discrepancies found
- MultinomialNB log scores: -7.51→-10.12, -11.94→-14.71
- Correlated features accuracy: 98.3%→93.3%
- Scenario 1 (small data): 80.0%→66.7%
- Scenario 3 (large+corr): 97.8%→82.2%
- Scenario 4 (multi-class): 88.3%→58.3%
