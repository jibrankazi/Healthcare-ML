"""External-cohort evaluation gate. Never train, tune, or select models here.

A caller-provided CSV is NOT independent clinical evidence merely because it
passes this gate: provenance, collection site, and patient separation must be
independently audited by the data custodian.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)

FEATURES = list(load_breast_cancer().feature_names)


def signatures(frame):
    arr = frame.loc[:, FEATURES].to_numpy(dtype="<f8")
    return {hashlib.sha256(row.tobytes()).hexdigest() for row in arr}


def evaluate_external(csv_path, model_path, *, source_url, collection_site,
                      collection_period, target="diagnosis", patient_id="patient_id",
                      positive_label="M"):
    if not (source_url.startswith("https://") and collection_site.strip()
            and collection_period.strip()):
        raise ValueError("HTTPS source URL, site, and collection period are required")
    frame = pd.read_csv(csv_path, low_memory=False)
    required = set(FEATURES) | {target, patient_id}
    if not required.issubset(frame):
        raise ValueError("External cohort missing required feature, diagnosis or patient ID columns")
    if len(frame) < 20 or frame[patient_id].isna().any() or frame[patient_id].duplicated().any():
        raise ValueError("External cohort requires >=20 records and unique nonmissing patient IDs")
    labels = frame[target].astype(str).str.strip()
    classes = set(labels)
    if len(classes) != 2 or positive_label not in classes:
        raise ValueError("Expected exactly two explicitly mapped diagnosis classes")
    y = labels.eq(positive_label).astype(int).to_numpy()
    if np.count_nonzero(y) < 5 or np.count_nonzero(1-y) < 5:
        raise ValueError("At least five cases of each class required")
    raw = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite external predictor values")
    # Reject a renamed or rehosted version of any row in the source WDBC cohort.
    original = load_breast_cancer(as_frame=True).data.loc[:, FEATURES]
    overlap = signatures(raw) & signatures(original)
    if overlap:
        raise ValueError("External records overlap original Wisconsin development cohort")
    model = joblib.load(model_path)  # Only load a locally trusted project artifact.
    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != FEATURES:
            raise ValueError("Fitted model's feature schema differs from original 30 predictors")
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("Fitted probability classifier required")
    class_positions = list(model.classes_)
    if 1 not in class_positions:
        raise ValueError("Trained classifier must predict malignant=1")
    score = np.asarray(model.predict_proba(raw), dtype=float)[:, class_positions.index(1)]
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Model returned invalid probabilities")
    pred = score >= .5
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "source_url_as_supplied_not_independently_authenticated": source_url,
        "collection_site_as_supplied": collection_site,
        "collection_period_as_supplied": collection_period,
        "cohort_file_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "model_file_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "n_records": int(len(frame)), "malignant_positive": int(sum(y)),
        "benign_negative": int(len(y)-sum(y)), "positive_label_supplied": positive_label,
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "sensitivity_at_0_5": tp / (tp+fn), "specificity_at_0_5": tn/(tn+fp),
        "confusion_tn_fp_fn_tp": [tn, fp, fn, tp],
        "exact_feature_row_overlap_with_wdbc": 0,
        "source_site_patient_disjointness_independently_audited": False,
        "clinical_external_validation_established": False,
        "use_for_care_or_deployment": False,
        "caution": "Metrics are exploratory until an independent clinical data custodian verifies site, labels, sampling, patient independence and diagnostic feature equivalence."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--period", required=True)
    parser.add_argument("--target", default="diagnosis")
    parser.add_argument("--patient-id", default="patient_id")
    parser.add_argument("--positive-label", default="M")
    parser.add_argument("--output", default="runs/external_cohort_review.json")
    args = parser.parse_args()
    result = evaluate_external(args.csv, args.model, source_url=args.source_url,
                               collection_site=args.site, collection_period=args.period,
                               target=args.target, patient_id=args.patient_id,
                               positive_label=args.positive_label)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
