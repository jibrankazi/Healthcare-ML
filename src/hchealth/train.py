import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hchealth.data import load_tabular


def set_seed(s):
    random.seed(s)
    np.random.seed(s)


def main():
    ap = argparse.ArgumentParser(description="Train a clinical risk model")
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

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
            random_state=seed,
        )),
    ])
    model.fit(X_train, y_train)

    joblib.dump(model, out / "model.joblib")

    meta = {"n_features": int(X_train.shape[1]), "n_train": int(len(y_train))}
    with open(out / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()
