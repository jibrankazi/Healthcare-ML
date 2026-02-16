import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from hchealth.data import load_tabular


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained clinical risk model")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = Path(cfg["outputs"]["run_dir"])
    fig_dir = Path(cfg["outputs"]["fig_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(out / "model.joblib")

    X_train, X_test, y_train, y_test = load_tabular(
        cfg["data"]["hf_dataset_id"],
        cfg["data"]["target_column"],
    )

    proba = model.predict_proba(X_test)[:, 1]
    auroc = float(roc_auc_score(y_test, proba))
    auprc = float(average_precision_score(y_test, proba))

    results = {"AUROC": auroc, "AUPRC": auprc, "n_test": int(len(y_test))}
    with open(cfg["outputs"]["results_json"], "w") as f:
        json.dump(results, f, indent=2)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")
    ax.grid(True)
    fig.savefig(fig_dir / "roc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve")
    ax.grid(True)
    fig.savefig(fig_dir / "pr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    fig, ax = plt.subplots()
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("Mean predicted")
    ax.set_ylabel("Fraction positive")
    ax.set_title("Calibration")
    ax.grid(True)
    fig.savefig(fig_dir / "calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Results:", results)


if __name__ == "__main__":
    main()
