# Model Explainability with SHAP and LIME — Code from Blog Post

Extracted from: https://daddaops.com/blog/model-explainability-shap-lime/

## Scripts

| Script | Code Blocks | Description |
|--------|------------|-------------|
| `shapley_exact.py` | 1 | Exact Shapley values via brute-force enumeration for a 4-feature loan model |
| `shap_lime_demo.py` | 2-5 | TreeSHAP, LIME from scratch, stability comparison, production pipeline |

## Dependencies

- Python 3.8+
- numpy, scikit-learn, shap

## Running

```bash
pip install -r requirements.txt

# Block 1: Exact Shapley values (numpy only)
python shapley_exact.py

# Blocks 2-5: TreeSHAP, LIME, stability, pipeline
python shap_lime_demo.py
```
