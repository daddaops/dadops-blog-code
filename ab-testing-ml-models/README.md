# A/B Testing ML Models in Production

Verified, runnable code from the DadOps blog post:
[A/B Testing ML Models in Production](https://dadops.dev/blog/ab-testing-ml-models/)

## Scripts

- `statistical_tests.py` — Two-sample z-test for proportions, Welch's t-test for continuous metrics, and sample size calculator with tradeoff table
- `sequential_and_bandits.py` — O'Brien-Fleming sequential testing boundaries, A/A peeking simulation (10K runs), and Thompson Sampling bandit vs fixed 50/50 split
- `segments_and_pipeline.py` — Segment-level analysis with Bonferroni correction, guardrail checks, and complete ABTestRunner class with fraud detection simulation

## Usage

```bash
pip install -r requirements.txt
python statistical_tests.py
python sequential_and_bandits.py
python segments_and_pipeline.py
```
