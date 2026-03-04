"""Evaluate trained models: metrics, bootstrap CIs, SHAP, and comparison plots.

Generates:
  - Per-model AUROC / AUPRC with 95% bootstrap confidence intervals
  - Multi-model ROC overlay
  - Multi-model PR overlay
  - Multi-model calibration overlay
  - SHAP summary (beeswarm) plot for the best model
  - results.json with all metrics
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from hchealth.data import load_tabular

# Consistent colour palette across all plots
COLORS = {
    "LogReg": "#1f77b4",
    "RF": "#ff7f0e",
    "RF+Isotonic": "#2ca02c",
    "RF+Platt": "#d62728",
    "XGBoost": "#9467bd",
}


def bootstrap_metric(y_true, y_score, metric_fn, n_boot=1000, seed=42):
    """Return (point_estimate, ci_lower, ci_upper) via percentile bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_score)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            scores[i] = metric_fn(y_true[idx], y_score[idx])
        except ValueError:
            scores[i] = np.nan
    lo, hi = np.nanpercentile(scores, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained clinical risk models")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = Path(cfg["outputs"]["run_dir"])
    fig_dir = Path(cfg["outputs"]["fig_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_boot = cfg.get("cv", {}).get("n_bootstrap", 1000)
    seed = cfg.get("seed", 42)

    X_train, X_test, y_train, y_test = load_tabular(
        cfg["data"]["hf_dataset_id"],
        cfg["data"]["target_column"],
    )

    # Convert to numpy for bootstrap indexing
    y_test_np = np.array(y_test)

    # Load metadata to get model names
    with open(out / "train_meta.json") as f:
        meta = json.load(f)
    model_names = meta["models"]

    # --- Compute per-model metrics ---
    all_results = {}
    model_probas = {}

    for name in model_names:
        model = joblib.load(out / f"model_{name}.joblib")
        proba = model.predict_proba(X_test)[:, 1]
        model_probas[name] = proba

        auroc, auroc_lo, auroc_hi = bootstrap_metric(
            y_test_np, proba, roc_auc_score, n_boot, seed
        )
        auprc, auprc_lo, auprc_hi = bootstrap_metric(
            y_test_np, proba, average_precision_score, n_boot, seed
        )
        brier = float(brier_score_loss(y_test, proba))

        all_results[name] = {
            "AUROC": auroc,
            "AUROC_CI": [auroc_lo, auroc_hi],
            "AUPRC": auprc,
            "AUPRC_CI": [auprc_lo, auprc_hi],
            "Brier": brier,
        }
        print(f"  {name:15s}  AUROC={auroc:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]  "
              f"AUPRC={auprc:.4f}  Brier={brier:.4f}")

    results_out = {
        "n_test": int(len(y_test)),
        "n_bootstrap": n_boot,
        "models": all_results,
    }
    with open(cfg["outputs"]["results_json"], "w") as f:
        json.dump(results_out, f, indent=2)

    # --- Multi-model ROC ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in model_names:
        fpr, tpr, _ = roc_curve(y_test, model_probas[name])
        auroc = all_results[name]["AUROC"]
        ax.plot(fpr, tpr, color=COLORS.get(name, "gray"),
                label=f"{name} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / "roc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Multi-model PR ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in model_names:
        prec, rec, _ = precision_recall_curve(y_test, model_probas[name])
        auprc = all_results[name]["AUPRC"]
        ax.plot(rec, prec, color=COLORS.get(name, "gray"),
                label=f"{name} (AUPRC={auprc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / "pr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Multi-model Calibration ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in model_names:
        frac_pos, mean_pred = calibration_curve(
            y_test, model_probas[name], n_bins=10
        )
        brier = all_results[name]["Brier"]
        ax.plot(mean_pred, frac_pos, marker="o", markersize=4,
                color=COLORS.get(name, "gray"),
                label=f"{name} (Brier={brier:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves — Model Comparison")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / "calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- SHAP feature importance (best model by AUROC) ---
    best_name = max(all_results, key=lambda k: all_results[k]["AUROC"])
    print(f"\nGenerating SHAP analysis for best model: {best_name}")
    best_model = joblib.load(out / f"model_{best_name}.joblib")

    # Get the classifier from the pipeline and the scaled data
    scaler = best_model.named_steps["scaler"]
    clf = best_model.named_steps["clf"]
    X_test_scaled = scaler.transform(X_test)

    # Use TreeExplainer for tree models, KernelExplainer otherwise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(clf, "estimators_") or hasattr(clf, "get_booster"):
            # Tree-based model
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test_scaled)
            # For binary classification, TreeExplainer may return list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # class 1
        elif hasattr(clf, "calibrated_classifiers_"):
            # CalibratedClassifierCV — explain the base estimator
            base = clf.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(base)
            shap_values = explainer.shap_values(X_test_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Fallback: KernelExplainer (slower but universal)
            bg = shap.sample(X_test_scaled, min(50, len(X_test_scaled)))
            explainer = shap.KernelExplainer(clf.predict_proba, bg)
            shap_values = explainer.shap_values(X_test_scaled)[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_test_scaled,
        feature_names=list(X_test.columns),
        show=False, max_display=15,
    )
    plt.title(f"SHAP Feature Importance — {best_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    # --- Bar chart: top 15 features by mean |SHAP| ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = list(X_test.columns)
    sorted_idx = np.argsort(mean_abs_shap)[-15:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs_shap[sorted_idx],
        color="#1f77b4",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top 15 Feature Importances — {best_name}")
    ax.grid(True, alpha=0.3, axis="x")
    fig.savefig(fig_dir / "shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll plots saved to {fig_dir}/")
    print("Results:", json.dumps(
        {k: f"{v['AUROC']:.4f}" for k, v in all_results.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()
