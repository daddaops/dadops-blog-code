# Survival Analysis from Scratch

Code extracted from the DadOps blog post "Survival Analysis from Scratch: Modeling Time-to-Event Data with Censoring".

## Scripts

1. **01_weibull_functions.py** - Weibull hazard, cumulative hazard, and survival functions. Verifies the S(t) = exp(-H(t)) identity and shows how shape parameter k controls hazard behavior (decreasing, constant, increasing).

2. **02_kaplan_meier.py** - Kaplan-Meier estimator with Greenwood standard errors. Demonstrates non-parametric survival curve estimation on a 10-patient example with censoring.

3. **03_log_rank_test.py** - Two-sample log-rank test comparing survival between treatment and control groups. Uses chi-squared test statistic to detect significant differences.

4. **04_cox_proportional_hazards.py** - Cox Proportional Hazards model fitted via gradient descent, plus Breslow baseline hazard estimator. Recovers true coefficients from simulated data with censoring.

5. **05_concordance_index.py** - Harrell's C-index for evaluating survival model discrimination. Fits a Cox model and computes concordance on simulated data.

6. **06_weibull_mle.py** - Maximum likelihood estimation for Weibull parameters with censored data using Nelder-Mead optimization. Estimates shape and scale from simulated survival times.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python 01_weibull_functions.py
python 02_kaplan_meier.py
python 03_log_rank_test.py
python 04_cox_proportional_hazards.py
python 05_concordance_index.py
python 06_weibull_mle.py
```
