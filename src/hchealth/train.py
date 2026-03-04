"""Train multiple classifiers with cross-validation and optional calibration.

Models trained:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. Random Forest + Isotonic calibration
  4. Random Forest + Platt (sigmoid) calibration
  5. XGBoost

All models are wrapped in a StandardScaler pipeline and evaluated with
stratified k-fold cross-validation.  Final models are refit on the full
training set and serialised to disk.
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from hchealth.data import load_tabular


def set_seed(s):
    random.seed(s)
    np.random.seed(s)


def build_models(cfg):
    """Return an ordered dict of (name -> sklearn Pipeline)."""
    seed = cfg.get("seed", 42)
    n_est = cfg["model"]["n_estimators"]
    max_d = cfg["model"]["max_depth"]

    rf = RandomForestClassifier(
        n_estimators=n_est, max_depth=max_d, random_state=seed
    )

    models = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_est, max_depth=max_d, random_state=seed
            )),
        ]),
        "RF+Isotonic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(rf, method="isotonic", cv=3)),
        ]),
        "RF+Platt": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(rf, method="sigmoid", cv=3)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=n_est, max_depth=6,
                learning_rate=0.1, random_state=seed,
                eval_metric="logloss", verbosity=0,
            )),
        ]),
    }
    return models


def main():
    ap = argparse.ArgumentParser(description="Train clinical risk models")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    out = Path(cfg["outputs"]["run_dir"])
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_tabular(
        cfg["data"]["hf_dataset_id"],
        cfg["data"]["target_column"],
    )

    models = build_models(cfg)
    n_folds = cfg.get("cv", {}).get("n_folds", 5)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    cv_results = {}
    scoring = ["roc_auc", "average_precision"]

    for name, pipeline in models.items():
        print(f"  Cross-validating {name} ({n_folds}-fold)...")
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=scoring, return_train_score=False,
        )
        cv_results[name] = {
            "cv_auroc_mean": float(np.mean(scores["test_roc_auc"])),
            "cv_auroc_std": float(np.std(scores["test_roc_auc"])),
            "cv_auprc_mean": float(np.mean(scores["test_average_precision"])),
            "cv_auprc_std": float(np.std(scores["test_average_precision"])),
        }
        print(f"    AUROC = {cv_results[name]['cv_auroc_mean']:.4f} "
              f"+/- {cv_results[name]['cv_auroc_std']:.4f}")

    # Refit all models on full training set
    print("\nRefitting on full training set...")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, out / f"model_{name}.joblib")

    # Save metadata
    meta = {
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_folds": n_folds,
        "models": list(models.keys()),
        "cv_results": cv_results,
    }
    with open(out / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()
