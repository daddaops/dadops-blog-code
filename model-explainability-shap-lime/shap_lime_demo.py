"""
Code Blocks 2-5: TreeSHAP, LIME from Scratch, Stability Test, Production Pipeline.

From: https://dadops.dev/blog/model-explainability-shap-lime/

Blocks 2-5 share a trained GradientBoostingClassifier on synthetic loan data.
This script runs them all sequentially.

Dependencies: numpy, scikit-learn, shap
"""

import numpy as np
import shap
import json
import hashlib
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


# ─── Block 2: TreeSHAP with GradientBoostingClassifier ───

def run_block2():
    """Train model and explain with TreeSHAP."""
    print("=== Block 2: TreeSHAP Demo ===\n")

    # Generate synthetic loan data
    rng = np.random.RandomState(42)
    n_samples = 2000
    income = rng.normal(55000, 15000, n_samples)
    credit_score = rng.normal(680, 50, n_samples)
    debt_ratio = rng.uniform(0.1, 0.9, n_samples)
    employment_years = rng.exponential(5, n_samples).clip(0, 30)
    loan_amount = rng.normal(25000, 10000, n_samples).clip(5000, None)

    # Approval rule (with some noise)
    approval_score = (0.3 * (income / 80000) + 0.25 * (credit_score / 800)
                      - 0.35 * debt_ratio + 0.1 * (employment_years / 15))
    approved = (approval_score + rng.normal(0, 0.05, n_samples)) > 0.35

    X = np.column_stack([income, credit_score, debt_ratio,
                         employment_years, loan_amount])
    feature_names = ["income", "credit_score", "debt_ratio",
                     "employment_years", "loan_amount"]
    y = approved.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

    # Explain with TreeSHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Single prediction explanation
    idx = 0  # First test instance
    print(f"\nPrediction for applicant #{idx}: "
          f"{'Approved' if model.predict(X_test[idx:idx+1])[0] else 'Rejected'}")
    print(f"Base value (avg prediction): {shap_values[idx].base_values:.3f}")
    print(f"\nFeature contributions:")
    for name, val in zip(feature_names, shap_values[idx].values):
        direction = "toward approval" if val > 0 else "toward rejection"
        print(f"  {name:<20s} {val:+.4f}  ({direction})")

    return model, X_train, X_test, y_test, feature_names


# ─── Block 3: LIME from Scratch ───

def lime_explain(model_predict, instance, training_data,
                 n_samples=1000, kernel_width=0.75, random_state=42):
    """Explain a single prediction using LIME from scratch."""
    rng = np.random.RandomState(random_state)
    n_features = len(instance)

    # Step 1: Generate binary perturbation vectors
    perturbations = rng.randint(0, 2, size=(n_samples, n_features))
    perturbations[0] = np.ones(n_features)  # Always include the original

    # Step 2: Build actual samples from perturbation masks
    background = training_data.mean(axis=0)
    samples = np.where(perturbations, instance, background)

    # Step 3: Get model predictions on perturbed samples
    predictions = model_predict(samples)

    # Step 4: Compute proximity weights (exponential kernel)
    distances = np.sqrt(((perturbations - 1) ** 2).sum(axis=1))
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    # Step 5: Fit weighted linear regression
    local_model = Ridge(alpha=1.0)
    local_model.fit(perturbations, predictions, sample_weight=weights)

    return local_model.coef_, local_model.intercept_


def run_block3(model, X_train, X_test, feature_names):
    """Run LIME from scratch on same model."""
    print("\n=== Block 3: LIME from Scratch Demo ===\n")

    applicant = X_test[0]
    coefs, intercept = lime_explain(
        model_predict=lambda x: model.predict_proba(x)[:, 1],
        instance=applicant,
        training_data=X_train,
        n_samples=2000
    )

    print("LIME feature importances (local linear coefficients):")
    for name, coef in zip(feature_names, coefs):
        direction = "toward approval" if coef > 0 else "toward rejection"
        print(f"  {name:<20s} {coef:+.4f}  ({direction})")


# ─── Block 4: LIME Stability vs SHAP Stability ───

def run_block4(model, X_train, X_test, feature_names):
    """Compare LIME stability across seeds vs deterministic TreeSHAP."""
    print("\n=== Block 4: Stability Test ===\n")

    # LIME stability: run 10 times with different seeds
    lime_rankings = defaultdict(list)
    applicant = X_test[0]

    for seed in range(10):
        coefs, _ = lime_explain(
            model_predict=lambda x: model.predict_proba(x)[:, 1],
            instance=applicant,
            training_data=X_train,
            n_samples=2000,
            random_state=seed
        )
        ranking = np.argsort(np.abs(coefs))[::-1]
        for rank, feat_idx in enumerate(ranking):
            lime_rankings[feature_names[feat_idx]].append(rank + 1)

    print("LIME rank across 10 runs (lower = more important):")
    for name in feature_names:
        ranks = lime_rankings[name]
        print(f"  {name:<20s} ranks: {ranks}  "
              f"mean: {np.mean(ranks):.1f}  std: {np.std(ranks):.2f}")

    # SHAP stability: always the same
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_test[0:1])
    shap_ranking = np.argsort(np.abs(shap_vals.values[0]))[::-1]
    print(f"\nSHAP ranking (deterministic, run it 1000 times \u2014 same result):")
    for rank, idx in enumerate(shap_ranking):
        print(f"  #{rank+1}: {feature_names[idx]:<20s} "
              f"SHAP = {shap_vals.values[0][idx]:+.4f}")


# ─── Block 5: Production Explanation Pipeline ───

class ExplanationPipeline:
    def __init__(self, model, feature_names, background_data):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self.background_stats = {
            name: {"mean": bg_mean, "std": bg_std}
            for name, bg_mean, bg_std in zip(
                feature_names,
                background_data.mean(axis=0),
                background_data.std(axis=0))
        }
        self.cache = {}

    def _cache_key(self, instance):
        return hashlib.md5(instance.tobytes()).hexdigest()

    def explain(self, instance, audience="technical"):
        key = self._cache_key(instance)
        if key not in self.cache:
            sv = self.explainer(instance.reshape(1, -1))
            self.cache[key] = {
                "shap_values": sv.values[0].tolist(),
                "base_value": float(sv.base_values[0]),
                "prediction": int(self.model.predict(
                    instance.reshape(1, -1))[0]),
                "probability": float(self.model.predict_proba(
                    instance.reshape(1, -1))[0, 1])
            }

        result = self.cache[key]
        if audience == "technical":
            return self._format_technical(result, instance)
        return self._format_business(result, instance)

    def _format_technical(self, result, instance):
        return {
            "prediction": result["prediction"],
            "probability": round(result["probability"], 4),
            "base_value": round(result["base_value"], 4),
            "features": {
                name: {"value": float(instance[i]),
                       "shap": round(result["shap_values"][i], 4)}
                for i, name in enumerate(self.feature_names)
            }
        }

    def _format_business(self, result, instance):
        pairs = list(zip(self.feature_names, result["shap_values"]))
        top = sorted(pairs, key=lambda p: abs(p[1]), reverse=True)[:3]
        decision = "approved" if result["prediction"] == 1 else "rejected"
        reasons = []
        for name, sv in top:
            idx = self.feature_names.index(name)
            val = instance[idx]
            avg = self.background_stats[name]["mean"]
            direction = "above" if val > avg else "below"
            reasons.append(
                f"{name} ({val:.0f}) is {direction} average ({avg:.0f})")
        return {
            "decision": decision,
            "confidence": f"{result['probability']:.0%}",
            "top_reasons": reasons
        }


def run_block5(model, X_train, X_test, feature_names):
    """Run the production explanation pipeline."""
    print("\n=== Block 5: Production Explanation Pipeline ===\n")

    pipeline = ExplanationPipeline(model, feature_names, X_train)
    applicant = X_test[0]

    tech_explanation = pipeline.explain(applicant, audience="technical")
    print("Technical:\n", json.dumps(tech_explanation, indent=2))

    biz_explanation = pipeline.explain(applicant, audience="business")
    print("\nBusiness:\n", json.dumps(biz_explanation, indent=2))


if __name__ == "__main__":
    model, X_train, X_test, y_test, feature_names = run_block2()
    run_block3(model, X_train, X_test, feature_names)
    run_block4(model, X_train, X_test, feature_names)
    run_block5(model, X_train, X_test, feature_names)
