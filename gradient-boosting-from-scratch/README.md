# Gradient Boosting from Scratch

Verified, runnable code from the [Gradient Boosting from Scratch](https://dadops.dev/blog/gradient-boosting-from-scratch/) blog post.

## Scripts

- **decision_tree.py** — Decision tree classifier with Gini impurity
- **random_forest.py** — Random forest via bagging + feature subsampling
- **gradient_boosting.py** — Gradient boosting regressor (MSE loss)
- **xgboost_tree.py** — XGBoost-style tree with second-order gradients
- **histogram_splitting.py** — Histogram-based splitting (LightGBM-style)

## Run

```bash
pip install -r requirements.txt
python decision_tree.py
python random_forest.py
python gradient_boosting.py
python xgboost_tree.py
python histogram_splitting.py
```
