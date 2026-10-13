"""Auditable research-only inference for a locally trained Wisconsin model.

This module deliberately refuses to infer with unknown feature schemas, nonfinite
input or mislabeled model metadata. Only load artifacts that were produced by this
trusted local project; joblib/pickle is unsafe for untrusted files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer

from hchealth.evaluate import choose_explanation_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_inference(
    config_path: str | Path,
    input_csv: str | Path,
    output_path: str | Path,
) -> dict:
    """Score a feature-only CSV using a training-CV-selected fitted model."""
    config_path = Path(config_path)
    input_csv = Path(input_csv)
    output_path = Path(output_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["data"]["hf_dataset_id"] != "none":
        raise ValueError("Inference currently supports only explicit Wisconsin research data")
    source = load_breast_cancer(as_frame=True)
    expected = list(source.feature_names)
    meta_file = Path(cfg["outputs"]["run_dir"]) / "train_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if (
        meta.get("data_source") != "sklearn Wisconsin Breast Cancer"
        or meta.get("positive_label") != 1
        or meta.get("positive_outcome") != "malignant"
        or meta.get("n_features") != len(expected)
    ):
        raise ValueError("Model metadata lacks required source or malignant-positive contract")
    model_name = choose_explanation_model(meta)
    model_path = meta_file.parent / f"model_{model_name}.joblib"

    # Reject target/outcome columns and extra feature names: no silent reorder,
    # accidental loss of predictors or inference on wrong label conventions.
    features = pd.read_csv(input_csv)
    if features.empty or list(features.columns) != expected:
        raise ValueError("Input must contain exactly 30 original ordered Wisconsin feature columns")
    if features.shape[0] > 10000:
        raise ValueError("Inference batch exceeds bounded research capacity")
    if not all(pd.api.types.is_numeric_dtype(features[c]) for c in expected):
        raise ValueError("All 30 predictor columns must be numerical")
    values = features.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictor values are prohibited")

    # Only trust locally produced training artifacts. Do NOT load an uploaded
    # joblib file or pass user-controlled arbitrary model paths to this CLI.
    model = joblib.load(model_path)
    model_columns = getattr(model, "feature_names_in_", None)
    if model_columns is not None and list(model_columns) != expected:
        raise ValueError("Fitted model and original feature ordering disagree")
    if not hasattr(model, "predict_proba") or 1 not in list(model.classes_):
        raise ValueError("Fitted model cannot generate malignant-positive probabilities")
    col = list(model.classes_).index(1)
    scores = np.asarray(model.predict_proba(features), dtype=float)[:, col]
    if scores.shape != (len(features),) or not np.isfinite(scores).all():
        raise ValueError("Fitted model returned invalid batch probability shape/values")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Fitted model returned probabilities outside 0..1")

    result = {
        "scope": "Offline Wisconsin Breast Cancer public-research inference ONLY",
        "clinical_decision_support": False,
        "external_clinical_validation_established": False,
        "positive_label": 1,
        "positive_outcome": "malignant",
        "selected_model": model_name,
        "model_selection_method": "training cross-validation AUROC; never heldout selection",
        "data_source": meta["data_source"],
        "input_sha256": sha256_file(input_csv),
        "model_sha256": sha256_file(model_path),
        "training_metadata_sha256": sha256_file(meta_file),
        "n_rows": len(features),
        "n_features": len(expected),
        "feature_names": expected,
        "probabilities_malignant": [float(x) for x in scores],
        "warning": (
            "Not validated for clinical decisions. Probabilities describe a "
            "historic research dataset distribution, NOT a diagnosis, prognosis "
            "or calibrated risk estimate for patients."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ci_demo.yaml")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", default="runs/ci_demo/research_inference.json")
    args = parser.parse_args()
    result = run_inference(args.config, args.input_csv, args.output)
    print(json.dumps({
        "selected_model": result["selected_model"],
        "n_rows": result["n_rows"],
        "output": str(args.output),
        "clinical_decision_support": False,
    }, indent=2))


if __name__ == "__main__":
    main()
